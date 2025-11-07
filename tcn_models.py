import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import total_anomaly_vae_loss


# =========================
# Core TCN Components
# =========================

class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network (TCN) with dilated convolutions
    """
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, 
                                   stride=1, dilation=dilation_size, 
                                   padding=(kernel_size-1) * dilation_size, 
                                   dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x if self.chomp_size == 0 else x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


# =========================
# Multi-Scale TCN
# =========================


class ResidualFeedForward(nn.Module):
    """LayerNorm-MLP residual block used in encoder/decoder projections."""

    def __init__(self, dim: int, hidden_dim: int | None = None, dropout: float = 0.1):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.norm(x)
        out = self.fc1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        return residual + out

class MultiScaleTCNEncoder(nn.Module):
    """
    多尺度时间卷积网络编码器
    使用不同卷积核大小的并行TCN分支捕捉多时间尺度特征
    """
    
    def __init__(self, num_inputs, output_channels, kernel_sizes=[3, 5, 7], 
                 base_channels=32, num_levels=3, dropout=0.2):
        super(MultiScaleTCNEncoder, self).__init__()
        
        self.kernel_sizes = kernel_sizes
        self.num_scales = len(kernel_sizes)
        
        # 为每个尺度分配通道数 - 平均分配总通道数
        channels_per_scale = output_channels // self.num_scales
        remaining_channels = output_channels % self.num_scales
        
        # 创建多个TCN分支，每个分支使用不同的卷积核大小
        self.tcn_branches = nn.ModuleList()
        self.scale_channels = []
        
        for i, kernel_size in enumerate(kernel_sizes):
            # 分配通道数，将余数分配给前几个分支
            scale_output_channels = channels_per_scale + (1 if i < remaining_channels else 0)
            self.scale_channels.append(scale_output_channels)
            
            # 构建每个尺度的通道配置
            if num_levels == 1:
                tcn_channels = [scale_output_channels]
            else:
                # 渐进式增长：base -> scale_output_channels
                tcn_channels = []
                for level in range(num_levels):
                    channels = int(base_channels + (scale_output_channels - base_channels) * level / (num_levels - 1))
                    tcn_channels.append(channels)
                tcn_channels[-1] = scale_output_channels  # 确保最后一层输出正确的通道数
            
            # 创建TCN分支
            tcn_branch = TemporalConvNet(
                num_inputs=num_inputs,
                num_channels=tcn_channels,
                kernel_size=kernel_size,
                dropout=dropout
            )
            self.tcn_branches.append(tcn_branch)
        
        # 特征融合层
        total_concat_channels = sum(self.scale_channels)
        self.fusion_conv = nn.Conv1d(total_concat_channels, output_channels, kernel_size=1)
        self.fusion_ln = nn.LayerNorm(output_channels)
        self.fusion_relu = nn.ReLU()
        self.fusion_dropout = nn.Dropout(dropout)
        
        # 尺度注意力机制 - 学习不同尺度的重要性权重
        self.scale_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # 全局平均池化
            nn.Conv1d(total_concat_channels, self.num_scales, kernel_size=1),
            nn.Sigmoid()
        )
        
        print(f"MultiScaleTCNEncoder初始化:")
        print(f"  - 卷积核大小: {kernel_sizes}")
        print(f"  - 各尺度通道数: {self.scale_channels}")
        print(f"  - 总输出通道数: {output_channels}")
        print(f"  - TCN层数: {num_levels}")
    
    def forward(self, x):
        """
        多尺度前向传播
        Args:
            x: (batch_size, num_inputs, sequence_length) - 输入时序数据
        Returns:
            fused_output: (batch_size, output_channels, sequence_length) - 融合的多尺度特征
        """
        batch_size, num_inputs, seq_len = x.shape
        
        # 并行处理各个尺度
        scale_outputs = []
        for i, tcn_branch in enumerate(self.tcn_branches):
            # 每个分支独立处理输入
            scale_output = tcn_branch(x)  # (batch_size, scale_channels[i], seq_len)
            scale_outputs.append(scale_output)
        
        # 在通道维度上拼接所有尺度的输出
        concatenated = torch.cat(scale_outputs, dim=1)  # (batch_size, total_concat_channels, seq_len)
        
        # 计算尺度注意力权重
        attention_weights = self.scale_attention(concatenated)  # (batch_size, num_scales, 1)
        
        # 对每个尺度的输出应用注意力权重
        weighted_outputs = []
        start_idx = 0
        for i, scale_channels in enumerate(self.scale_channels):
            end_idx = start_idx + scale_channels
            scale_feature = concatenated[:, start_idx:end_idx, :]  # (batch_size, scale_channels, seq_len)
            
            # 应用注意力权重
            attention_weight = attention_weights[:, i:i+1, :]  # (batch_size, 1, 1)
            weighted_feature = scale_feature * attention_weight  # 广播乘法
            weighted_outputs.append(weighted_feature)
            
            start_idx = end_idx
        
        # 重新拼接加权后的特征
        weighted_concatenated = torch.cat(weighted_outputs, dim=1)
        
        # 特征融合
        fused = self.fusion_conv(weighted_concatenated)  # (batch_size, output_channels, seq_len)
        fused = fused.transpose(1, 2)  # (batch_size, seq_len, output_channels) for LayerNorm
        fused = self.fusion_ln(fused)
        fused = fused.transpose(1, 2)  # (batch_size, output_channels, seq_len) back to conv format
        fused = self.fusion_relu(fused)
        fused = self.fusion_dropout(fused)
        
        return fused
    
    def get_scale_features(self, x):
        """
        获取各个尺度的独立特征（用于分析和可视化）
        Returns:
            scale_features: list of tensors, 每个tensor对应一个尺度的特征
        """
        scale_features = []
        for tcn_branch in self.tcn_branches:
            scale_feature = tcn_branch(x)
            scale_features.append(scale_feature)
        return scale_features


# =========================
# TCN-VAE and Data Processing
# =========================

class TCNVAE(nn.Module):
    """
    Temporal Convolutional Network Variational Autoencoder (TCN-VAE)
    Processes windowed time series data X∈R(Tw×P)
    """
    
    def __init__(self, input_channels, window_size, latent_dim=64, tcn_channels=[32, 64, 128], 
                 kernel_size=3, dropout=0.2, beta=4.0, use_multiscale=True, 
                 multiscale_kernels=[3, 5, 7], use_attention=True, use_residual=True):
        super(TCNVAE, self).__init__()
        self.input_channels = input_channels  # P (number of variables/channels)
        self.window_size = window_size        # Tw (window length)
        self.latent_dim = latent_dim
        self.beta = beta
        self.use_multiscale = use_multiscale
        self.use_attention = use_attention
        self.use_residual = use_residual
        
        # 选择编码器类型
        if use_multiscale:
            # 多尺度TCN编码器
            print(f"使用多尺度TCN编码器，卷积核: {multiscale_kernels}")
            self.tcn_encoder = MultiScaleTCNEncoder(
                num_inputs=input_channels,
                output_channels=tcn_channels[-1],  # 使用最后一层的通道数作为输出
                kernel_sizes=multiscale_kernels,
                base_channels=tcn_channels[0] if len(tcn_channels) > 0 else 32,
                num_levels=len(tcn_channels),
                dropout=dropout
            )
        else:
            # 原始单尺度TCN编码器（向后兼容）
            print(f"使用单尺度TCN编码器，卷积核: {kernel_size}")
            self.tcn_encoder = TemporalConvNet(input_channels, tcn_channels, 
                                             kernel_size=kernel_size, dropout=dropout)
        
        # After TCN, we need to flatten and compute latent parameters
        # The output size depends on the input size and TCN architecture
        self.encoder_output_dim = tcn_channels[-1] * window_size
        
        # 添加注意力机制（可选）
        if use_attention:
            # 确保embed_dim能被num_heads整除
            embed_dim = tcn_channels[-1]
            num_heads = 4
            if embed_dim % num_heads != 0:
                num_heads = 2  # 降为2个头以兼容更多维度
                if embed_dim % num_heads != 0:
                    num_heads = 1  # 最后使用1个头
            
            self.encoder_attention = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(embed_dim)
        
        # Residual enhancement and dimensionality reduction for encoder features
        self.encoder_residual = ResidualFeedForward(self.encoder_output_dim, dropout=dropout)
        self.encoder_projection = nn.Sequential(
            nn.LayerNorm(self.encoder_output_dim),
            nn.Linear(self.encoder_output_dim, self.encoder_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(self.encoder_output_dim // 2),
            nn.Linear(self.encoder_output_dim // 2, self.encoder_output_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.encoder_feature_norm = nn.LayerNorm(self.encoder_output_dim // 4)
        
        # Latent space mapping - 使用更小的中间层
        feature_dim = self.encoder_output_dim // 4
        self.fc_mu = nn.Linear(feature_dim + 1, latent_dim)  # +1 for label y
        self.fc_logvar = nn.Linear(feature_dim + 1, latent_dim)
        
        # 潜在空间正则化
        self.latent_ln = nn.LayerNorm(latent_dim)
        
        # TCN Decoder - reverse the channel order
        decoder_channels = tcn_channels[::-1]
        
        # 增强解码器路径
        decoder_feature_dim = self.encoder_output_dim // 4
        self.decoder_residual = ResidualFeedForward(decoder_feature_dim, dropout=dropout)

        # Map from latent space back to feature space
        self.fc_decode_1 = nn.Linear(latent_dim + 1, decoder_feature_dim)  # +1 for label y

        # 特征重构层
        self.feature_reconstructor = nn.Sequential(
            nn.LayerNorm(decoder_feature_dim),
            nn.Linear(decoder_feature_dim, decoder_feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_feature_dim * 2, self.encoder_output_dim),
            nn.LayerNorm(self.encoder_output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # TCN Decoder
        self.tcn_decoder = TemporalConvNet(decoder_channels[0], 
                                         decoder_channels[1:] + [input_channels],
                                         kernel_size=kernel_size, dropout=dropout)
        
        # 解码器注意力（如果启用）
        if use_attention:
            # 确保embed_dim能被num_heads整除
            decoder_embed_dim = input_channels
            decoder_num_heads = 2
            if decoder_embed_dim % decoder_num_heads != 0:
                decoder_num_heads = 1  # 使用1个头以兼容任何维度
            
            self.decoder_attention = nn.MultiheadAttention(
                embed_dim=decoder_embed_dim,
                num_heads=decoder_num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.decoder_attention_norm = nn.LayerNorm(decoder_embed_dim)
        
        # 增强的输出层
        self.output_layers = nn.Sequential(
            nn.Linear(input_channels, input_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(input_channels * 2, input_channels),
            nn.Tanh()  # 有界输出
        )
        
        # 输出归一化
        self.output_norm = nn.LayerNorm(input_channels)
        
    def encode(self, x, y):
        """
        Encode windowed time series (x, y) into latent space with enhanced features
        x: (batch_size, input_channels, window_size) - windowed time series
        y: (batch_size, 1) or (batch_size,) - window-level or aggregated labels
        """
        # TCN encoding
        h = self.tcn_encoder(x)  # (batch_size, tcn_channels[-1], window_size)
        
        # 应用注意力机制（如果启用）
        if self.use_attention:
            # 转换为序列格式用于注意力
            h_seq = h.transpose(1, 2)  # (batch_size, window_size, tcn_channels[-1])
            
            # 应用多头注意力
            h_attended, attention_weights = self.encoder_attention(h_seq, h_seq, h_seq)
            
            # 残差连接和归一化
            if self.use_residual:
                h_attended = self.attention_norm(h_seq + h_attended)
            else:
                h_attended = self.attention_norm(h_attended)
            
            # 转换回原格式
            h = h_attended.transpose(1, 2)  # (batch_size, tcn_channels[-1], window_size)
        
        # Flatten TCN output - use reshape for DataParallel compatibility
        h_flat = h.reshape(h.size(0), -1)  # (batch_size, encoder_output_dim)
        h_flat = self.encoder_residual(h_flat)
        
        # 特征提取
        h_features = self.encoder_projection(h_flat)  # (batch_size, encoder_output_dim // 4)
        h_features = self.encoder_feature_norm(h_features)
        
        # Ensure y has correct dimensions
        if y.dim() == 1:
            y = y.unsqueeze(1)  # Convert (batch_size,) to (batch_size, 1)
        elif y.dim() > 2:
            y = y.view(y.size(0), -1)  # Flatten to (batch_size, -1)
            
        # Concatenate with label
        hy = torch.cat([h_features, y], dim=1)  # (batch_size, feature_dim + 1)
        
        # Compute latent parameters
        mu = self.fc_mu(hy)
        logvar = self.fc_logvar(hy)
        
        # 潜在空间正则化
        mu = self.latent_ln(mu)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Sample from latent space using reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, y):
        """
        Decode latent representation (z, y) back to time series with enhanced reconstruction
        z: (batch_size, latent_dim) - latent representation
        y: (batch_size, 1) or (batch_size,) - window-level labels
        """
        # Ensure y has correct dimensions
        if y.dim() == 1:
            y = y.unsqueeze(1)  # Convert (batch_size,) to (batch_size, 1)
        elif y.dim() > 2:
            y = y.view(y.size(0), -1)  # Flatten to (batch_size, -1)
            
        # Concatenate latent vector with label
        zy = torch.cat([z, y], dim=1)  # (batch_size, latent_dim + 1)
        
        # Map back to feature space
        h_features = self.fc_decode_1(zy)  # (batch_size, decoder_feature_dim)
        h_features = self.decoder_residual(h_features)

        # 特征重构
        h = self.feature_reconstructor(h_features)  # (batch_size, encoder_output_dim)
        
        # Reshape for TCN decoder
        h = h.view(h.size(0), -1, self.window_size)  # (batch_size, channels, window_size)
        
        # TCN decoding
        x_recon = self.tcn_decoder(h)  # (batch_size, input_channels, window_size)
        
        # 应用解码器注意力（如果启用）
        if self.use_attention:
            # 转换为序列格式
            x_seq = x_recon.transpose(1, 2)  # (batch_size, window_size, input_channels)
            
            # 应用注意力
            x_attended, _ = self.decoder_attention(x_seq, x_seq, x_seq)
            
            # 残差连接和归一化
            if self.use_residual:
                x_attended = self.decoder_attention_norm(x_seq + x_attended)
            else:
                x_attended = self.decoder_attention_norm(x_attended)
            
            # 转换回原格式
            x_recon = x_attended.transpose(1, 2)  # (batch_size, input_channels, window_size)
        
        # 应用增强的输出层
        x_recon = x_recon.transpose(1, 2)  # (batch_size, window_size, input_channels)
        x_recon = self.output_layers(x_recon)  # 有界输出
        
        # 输出归一化
        x_recon = self.output_norm(x_recon)
        
        x_recon = x_recon.transpose(1, 2)  # (batch_size, input_channels, window_size)
        
        return x_recon
    
    def get_multiscale_features(self, x):
        """
        获取多尺度特征（仅当使用多尺度编码器时可用）
        Args:
            x: (batch_size, input_channels, window_size) - 输入时序数据
        Returns:
            scale_features: list of tensors 或 None，每个tensor对应一个尺度的特征
        """
        if self.use_multiscale and hasattr(self.tcn_encoder, 'get_scale_features'):
            return self.tcn_encoder.get_scale_features(x)
        else:
            print("警告: 当前模型未使用多尺度编码器或编码器不支持特征分解")
            return None
    
    def get_encoder_info(self):
        """获取编码器配置信息"""
        if self.use_multiscale:
            return {
                'encoder_type': 'MultiScale',
                'kernel_sizes': getattr(self.tcn_encoder, 'kernel_sizes', []),
                'num_scales': getattr(self.tcn_encoder, 'num_scales', 0),
                'scale_channels': getattr(self.tcn_encoder, 'scale_channels', [])
            }
        else:
            return {
                'encoder_type': 'SingleScale',
                'kernel_size': getattr(self.tcn_encoder, 'kernel_size', None)
            }
    
    def forward(self, x, y):
        """Forward pass through the TCN-VAE"""
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, y)
        return x_recon, mu, logvar


class WindowDataProcessor:
    """
    Process time series data into windowed format
    Converts time series x1:T into windows X∈R(Tw×P)
    """
    
    def __init__(self, window_size, stride=1, overlap_ratio=0.5):
        self.window_size = window_size  # Tw
        self.stride = stride if stride > 0 else max(1, int(window_size * (1 - overlap_ratio)))
        
    def create_windows(self, x, y=None):
        """
        Create sliding windows from time series data
        Args:
            x: (batch_size, seq_len, n_features) or (seq_len, n_features) - time series
            y: (batch_size, seq_len) or (seq_len,) - point-level labels (optional)
        Returns:
            windowed_x: (n_windows, n_features, window_size)
            windowed_y: (n_windows, 1) - window-level labels
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension
        if y is not None and y.dim() == 1:
            y = y.unsqueeze(0)
            
        batch_size, seq_len, n_features = x.shape
        
        # Calculate number of windows
        n_windows = (seq_len - self.window_size) // self.stride + 1
        
        windowed_data = []
        windowed_labels = []
        
        for i in range(batch_size):
            for start_idx in range(0, seq_len - self.window_size + 1, self.stride):
                end_idx = start_idx + self.window_size
                
                # Extract window: (window_size, n_features) -> transpose to (n_features, window_size)
                window = x[i, start_idx:end_idx, :].transpose(0, 1)
                windowed_data.append(window)
                
                # Process labels if provided
                if y is not None:
                    window_labels = y[i, start_idx:end_idx]
                    # Window-level anomaly: 1 if any point in window is anomalous
                    window_label = torch.max(window_labels).float().unsqueeze(0)
                    windowed_labels.append(window_label)
                else:
                    windowed_labels.append(torch.tensor([0.0]))  # Default label
        
        windowed_x = torch.stack(windowed_data)  # (n_windows, n_features, window_size)
        windowed_y = torch.stack(windowed_labels)  # (n_windows, 1)
        
        return windowed_x, windowed_y
    
    def aggregate_point_labels(self, point_labels, aggregation='max'):
        """
        Aggregate point-level labels to window-level labels
        Args:
            point_labels: (window_size,) - point-level labels within a window
            aggregation: 'max', 'mean', 'any' - aggregation method
        """
        if aggregation == 'max':
            return torch.max(point_labels)
        elif aggregation == 'mean':
            return torch.mean(point_labels.float())
        elif aggregation == 'any':
            return torch.any(point_labels > 0).float()
        else:
            return torch.max(point_labels)  # Default to max


# =========================
# Transformer Models
# =========================

class PositionalEncoding(nn.Module):
    """
    Add positional encoding to the input for sequential modeling.
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # Add batch dimension

    def forward(self, x):
        """Add positional encoding to input tensor."""
        L = x.size(1)  # Sequence length
        return x + self.pe[:, :L, :].to(x.device)


class TransformerDetector(nn.Module):
    """
    Transformer-based detector model for anomaly or binary classification tasks.
    """

    def __init__(self, input_size, d_model=128, nhead=8, num_layers=2, dim_feedforward=256, dropout=0.1):
        super(TransformerDetector, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)  # Input embedding layer
        self.positional_encoding = PositionalEncoding(d_model)  # Positional encoding

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Fully connected layers for classification
        self.fc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output probability for binary classification
        )

    def forward(self, x):
        """Forward pass through the Transformer Detector."""
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension (B, 1, input_size)
        x = self.embedding(x)  # Project input to d_model dimensions
        x = self.positional_encoding(x)  # Add positional encoding
        x = self.transformer_encoder(x)  # Transformer encoder
        x = x.mean(dim=1)  # Aggregate features by averaging over sequence dimension
        return self.fc(x).squeeze(1)  # Output probabilities


# =========================
# Adversarial Sample Generation
# =========================

def TCN_One_Step_To_Feasible_Action(
        tcnvae,
        detector,
        x_orig,
        device,
        previously_generated=None,
        alpha=1.0,
        lambda_div=0.1,
        lr=0.001,
        steps=50,
        log_file=None,
        total_loss_weight=0.1,
        loss_alpha=1.0,
        loss_beta=1.0,
        loss_gamma=1.0,
        loss_zeta=1.0,
        loss_delta_min=0.1,
        loss_delta_max=1.0,
        loss_sigma_prior=0.5,
):
    """
    Generate adversarial windowed time series samples by modifying latent space representation.
    Args:
        tcnvae: Trained TCN-VAE model.
        detector: Trained detector model.
        x_orig: Original windowed time series data (channels, window_size).
        device: Computation device (CPU/GPU).
        previously_generated: List of previously generated samples (for diversity).
        alpha: Scaling factor for diversity term.
        lambda_div: Weight for diversity term.
        lr: Learning rate for optimization.
        steps: Number of optimization steps.
        log_file: Path to log file for recording progress.
    Returns:
        Adversarial windowed time series sample (torch.Tensor).
    """
    tcnvae.eval()
    detector.eval()

    if previously_generated is None:
        previously_generated = []

    # Ensure proper shape for TCN-VAE input
    if x_orig.dim() == 2:  # (channels, window_size)
        x_orig = x_orig.unsqueeze(0)  # Add batch dimension (1, channels, window_size)
    x_orig = x_orig.to(device)
    
    y_class1 = torch.full((1, 1), 0.8, device=device)  # Target class label

    # Encode windowed time series into latent space
    with torch.no_grad():
        mean, logvar = tcnvae.encode(x_orig, y_class1)

    mu = mean.detach()
    sigma = torch.exp(0.5 * logvar).detach()
    epsilon = torch.randn_like(sigma)
    with torch.no_grad():
        base_recon = tcnvae.decode(mu, y_class1).detach()

    psi_param = torch.zeros_like(sigma, requires_grad=True)
    optimizer_psi = torch.optim.Adam([psi_param], lr=lr)

    for step in range(steps):
        optimizer_psi.zero_grad()

        psi = torch.exp(psi_param)
        z = mu + psi * (sigma * epsilon)

        # Decode latent variable back to windowed time series
        x_synthetic = tcnvae.decode(z, y_class1)  # (1, channels, window_size)

        # For detector, we may need to flatten or process the windowed data
        # Assuming detector can handle windowed input, otherwise flatten:
        if hasattr(detector, 'forward'):
            # Check if detector expects flattened input
            try:
                prob_class1 = detector(x_synthetic)
            except:
                # If detector expects 2D input, flatten the windows
                x_flat = x_synthetic.view(x_synthetic.size(0), -1)  # (1, channels*window_size)
                prob_class1 = detector(x_flat)
        else:
            x_flat = x_synthetic.view(x_synthetic.size(0), -1)
            prob_class1 = detector(x_flat)

        # Diversity term (if previous samples exist)
        diversity_term = 0.0
        if previously_generated:
            # Stack previous samples and compute distance in flattened space
            x_synthetic_flat = x_synthetic.view(1, -1)  # Flatten current sample
            x_old_cat = torch.stack([x.view(-1) for x in previously_generated], dim=0).to(device)
            dist = torch.norm(x_synthetic_flat - x_old_cat, p=2, dim=1)  # Pairwise distances
            diversity_term = torch.exp(-alpha * dist).sum()

        # Calculate total reward (inverse objective)
        total_loss, recon_l, perturb_l, zero_l, kl_l = total_anomaly_vae_loss(
            x=x_orig,
            x_recon=base_recon,
            mu=mu,
            logvar=logvar,
            x_tilde=x_synthetic,
            alpha=loss_alpha,
            beta=loss_beta,
            gamma=loss_gamma,
            zeta=loss_zeta,
            delta_min=loss_delta_min,
            delta_max=loss_delta_max,
            sigma_prior=loss_sigma_prior,
        )

        inv_reward = prob_class1.mean() + lambda_div * diversity_term - total_loss_weight * total_loss
        inv_reward.backward()
        optimizer_psi.step()

    psi = torch.exp(psi_param).detach()
    z_final = mu + psi * (sigma * epsilon)

    detector_reward = float(prob_class1.item())
    diversity_reward_display = float(diversity_term.detach().cpu()) if isinstance(diversity_term, torch.Tensor) else float(diversity_term)
    sample_reward = float(inv_reward.item())
    total_loss_value = float(total_loss.detach().cpu())

    print(
        f"TCN Deceiving Detector Reward: {detector_reward:.4f}",
        f"Diversity reward: {diversity_reward_display:.4f}",
        f"Sample reward: {sample_reward:.4f}",
        f"Total loss: {total_loss_value:.4f}",
        f"Psi mean: {psi.mean().item():.4f}"
    )

    # Decode optimized latent variable back to windowed time series
    with torch.no_grad():
        x_adv = tcnvae.decode(z_final, y_class1).detach().cpu().squeeze(0)  # Remove batch dimension
    
    return x_adv
