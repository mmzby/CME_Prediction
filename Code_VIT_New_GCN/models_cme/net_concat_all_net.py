import torch
import torch.nn as nn
from models_cme.vit_new import vit_with_ProbSparse
from models_cme.GCN import GCN_with_cosine


class MLP(nn.Module):
    def __init__(self, in_features=12, hidden=512):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            # nn.Linear(hidden, 1)
        )
        self.conv1 = nn.Conv1d(in_channels=hidden, out_channels=2048, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=2048, out_channels=hidden, kernel_size=1)
        self.lnorm = nn.LayerNorm(hidden)
        self.fc = nn.Sequential(nn.Linear(512 * 12, 512))

    def forward(self, data):
        shape = data.shape
        data = data.unsqueeze(-1)  # (4,12,1)
        # print(data.shape)
        data = self.net(data)  # (4,12,512)
        # print(data.shape)
        out = self.conv1(data.transpose(1, 2))  # (4,2048,12)
        # print(out.shape)
        out = self.conv2(out).transpose(1, 2)  # (4,12,512)
        # print(out.shape)
        out = self.lnorm(out + data)  # (4,12,512)
        # print(out.shape)
        out = out.squeeze(-1).reshape(shape[0], 12 * 512)
        out = self.fc(out)
        return out


class GCN_FusionModule(nn.Module):
    def __init__(self, para_dim, num_classes, num_heads=8, dim=64):
        """
        Fusion Module.

        Args:
            para_dim (int): Dimension of the input 1D parameters.
            output_dim (int): Dimension of the fused output feature.
        """
        super(GCN_FusionModule, self).__init__()

        # Load pretrained ResNet50 model
        # self.resnet50 = resnet.resnet50(num_classes=1)
        self.vit1 = vit_with_ProbSparse.vit_base_patch16_224_in21k(num_classes=1)
        self.vit2 = vit_with_ProbSparse.vit_base_patch16_224_in21k(num_classes=1)

        # 初始化 MLP
        self.mlp = MLP(in_features=para_dim, hidden=512)

        self.gcn_img = GCN_with_cosine(nfeat=256, nhid=512, nclass=256, dropout=0)
        self.gcn_pca = GCN_with_cosine(nfeat=256, nhid=512, nclass=256, dropout=0)
        self.gcn_para = GCN_with_cosine(nfeat=256, nhid=512, nclass=256, dropout=0)

        self.fc_gcn_img = nn.Sequential(nn.Linear(768, 256),
                                        nn.ReLU())  # Leaky

        self.fc_gcn_para = nn.Sequential(nn.Linear(512, 256),
                                         nn.ReLU())  # Leaky 

        # 最优11.202_14.156
        self.fusion_fc = nn.Sequential(nn.Linear(256*3, 256),  # 输入数据大小为 (batch_size, para_dim + 32)
                                       nn.ReLU(),
                                       nn.Linear(256, 128),  # 输入数据大小为 (batch_size, para_dim + 32)
                                       nn.ReLU())

        self.fusion_fc_1 = nn.Sequential(nn.Linear(128, num_classes))

    def forward(self, image, pca_image, para):
        """
        Forward pass.

        Args:
            image (Tensor): Input image tensor of shape (batch_size, 3, H, W).
            para (Tensor): Input 1D parameter tensor of shape (batch_size, para_dim).

        Returns:
            Tensor: Fused feature tensor of shape (batch_size, output_dim).
        """
        # Process image through ResNet50 to obtain a 1D vector
        # _, image_features = self.resnet50(image)  # Shape: (batch_size, 32)
        image_pre, image_features = self.vit1(image)  # (B, N)
        # _, image_features=self.resnet50(image)
        pca_image_pre, pca_image_features = self.vit2(pca_image)

        # 通过 MLP 处理 para 数据
        para_features = self.mlp(para)

        image_features = self.fc_gcn_img(image_features)
        pca_image_features = self.fc_gcn_img(pca_image_features)
        para_features = self.fc_gcn_para(para_features)

        gcn_para_features = self.gcn_para(para_features) + para_features
        gcn_image_features = self.gcn_img(image_features) + image_features
        gcn_pca_image_features = self.gcn_pca(pca_image_features) + pca_image_features

        # Concatenate image and parameter features
        fused_features = torch.cat((gcn_para_features,
                                    gcn_image_features,
                                    gcn_pca_image_features), dim=1)  
        
        # Fuse features through a fully connected layer
        output = self.fusion_fc(fused_features)  # Shape: (batch_size, output_dim)
        output = self.fusion_fc_1(output)

        return output


# Example usage
if __name__ == "__main__":
    # Define input dimensions
    # Initialize the fusion module
    fusion_module = GCN_FusionModule(para_dim=12, num_classes=1)

    # Example input data
    image = torch.randn(4, 2, 3, 224, 224)  # Batch of 8 images, 3x224x224
    pca_image = torch.randn(4, 2, 3, 224, 224)
    para = torch.randn(4, 12)  # Batch of 8 sets of 10-dimensional parameters

    # Forward pass
    output = fusion_module(image, pca_image, para)
    print("Output shape:", output.shape)
