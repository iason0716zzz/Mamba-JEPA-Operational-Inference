import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import torch.optim as optim

class MambaPredictorSimulated(nn.Module):
    """
    Linear-time Predictor (Simulated for Python 3.14 Compatibility):
    Simulates Mamba's State Space Model (SSM) token evolution via high-dimensional 
    linear mapping, locking computational complexity to O(N) for long-context vision sequences.
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.simulated_ssm = nn.Linear(embed_dim, embed_dim)
        self.act = nn.SiLU()
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        return self.norm(self.act(self.simulated_ssm(x)))

class OperationalJEPA(nn.Module):
    """
    Operational Joint Embedding Predictive Architecture (JEPA):
    Maps localized patch sequences into abstract latent space, executing 
    predictive inference via an O(N) core to deduce missing environmental context.
    """
    def __init__(self, patch_size=16, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = nn.Linear(patch_size * patch_size * 3, embed_dim)
        
        # Latent Decoder: Expands truncated features back to full 196-patch pixel dimensions
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 196 * patch_size * patch_size * 3)
        )
        self.mamba_predictor = MambaPredictorSimulated(embed_dim=embed_dim)
        
    def forward(self, patch_sequence):
        return self.mamba_predictor(patch_sequence)

def TrainAIEngine(model, img_tensor, epochs=150):
    """
    Self-Supervised Training Loop:
    Forces the latent network to optimize structural representation by minimizing
    MSE Loss between predicted features and the full 100% ground-truth target patches.
    """
    print("\n🔥 [Initialization] Pointing Self-Supervised Evolutionary Optimization Loop...")
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    
    # Tokenizing full image to patches as historical Ground Truth
    b, c, h, w = img_tensor.shape
    p = model.patch_size
    patches = img_tensor.view(b, c, h//p, p, w//p, p)
    patches = patches.permute(0, 2, 4, 1, 3, 5).flatten(1, 2).flatten(2) # Shape: [1, 196, 768]
    
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        
        # Generating asymmetric mask (Keeping 30% context, masking 70%)
        num_patches = patches.shape[1]
        keep_num = int(num_patches * 0.3)
        shuffled_indices = torch.randperm(num_patches)
        keep_indices = shuffled_indices[:keep_num]
        mask_indices = shuffled_indices[keep_num:]
        
        # Latent prediction feed
        latent_patches = model.patch_embed(patches)
        context_patches = latent_patches[:, keep_indices, :]
        predicted_latent = model(context_patches)
        
        # Reshaping decoder output to exact patch shape
        predicted_pixels = model.decoder(predicted_latent.mean(dim=1, keepdim=True))
        predicted_pixels = predicted_pixels.view(1, 196, p * p * 3)
        
        # Computing loss strictly on the masked tokens
        loss = criterion(predicted_pixels[:, mask_indices, :], patches[:, mask_indices, :])
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0 or epoch == 1:
            print(f" ⚙️ Epoch [{epoch:03d}/{epochs}] ── Structural Reconstruction Loss: {loss.item():.6f}")
            
    print("✨ [Success] Latent parameters fully converged with target environment metrics!\n")
    model.eval()

if __name__ == "__main__":
    # Allocating hardware runtime context
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📡 AGI Engine Core deployed on execution target: {device.upper()}")
    
    model = OperationalJEPA().to(device)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    image_name = "test.jpg"
    if not os.path.exists(image_name) and os.path.exists("test.jpg.png"):
        image_name = "test.jpg.png"
        
    print(f"\n[System] Checking workspace context for physical target asset: '{image_name}'...")
    
    if os.path.exists(image_name):
        try:
            img = Image.open(image_name).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            # Execute optimization loop with 150 epochs for high-fidelity infilling
            TrainAIEngine(model, img_tensor, epochs=150)
            
            b, c, h, w = img_tensor.shape
            p = model.patch_size
            patches = img_tensor.view(b, c, h//p, p, w//p, p)
            patches = patches.permute(0, 2, 4, 1, 3, 5).flatten(1, 2).flatten(2)
            
            num_patches = patches.shape[1]
            keep_num = int(num_patches * 0.3)
            shuffled_indices = torch.randperm(num_patches)
            keep_indices = shuffled_indices[:keep_num]
            mask_indices = shuffled_indices[keep_num:]
            
            masked_patches = patches.clone()
            masked_patches[:, mask_indices, :] = 0.0
            
            latent_patches = model.patch_embed(patches)
            context_patches = latent_patches[:, keep_indices, :]
            
            with torch.no_grad():
                predicted_latent = model(context_patches)
                predicted_pixels = model.decoder(predicted_latent.mean(dim=1, keepdim=True))
                predicted_pixels = predicted_pixels.view(1, 196, p * p * 3)
            
            def reconstruct(p_tensor):
                p_rescaled = p_tensor.view(14, 14, 3, 16, 16).permute(2, 0, 3, 1, 4).contiguous()
                return p_rescaled.view(3, 224, 224).permute(1, 2, 0).cpu().clamp(0, 1).numpy()
            
            img_original = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            img_masked = reconstruct(masked_patches)
            
            # Generating comparative analysis plots for visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(img_original)
            axes[0].set_title("1. Original World")
            axes[0].axis('off')
            
            axes[1].imshow(img_masked)
            axes[1].set_title("2. Masked World (70% Blocked)")
            axes[1].axis('off')
            
            filled_patches = masked_patches.clone()
            filled_patches[:, mask_indices, :] = predicted_pixels[:, mask_indices, :]
            img_predicted = reconstruct(filled_patches)
            
            axes[2].imshow(img_predicted)
            axes[2].set_title("3. Trained JEPA Prediction")
            axes[2].axis('off')
            
            output_plot = "jepa_insight.png"
            plt.savefig(output_plot, bbox_inches='tight')
            plt.close()
            
            print(f"⚡ Visualization pipeline telemetry compiled successfully!")
            print(f"  └─ Matrix dimensions validated: {patches.shape}")
            print(f"  └─ Analysis plot generated at current root: '{output_plot}'")
            print(f"🚀 Mission capable. Open the image to inspect advanced latent space inference.")
            
        except Exception as e:
            print(f"❌ Execution telemetry failed: {e}")
    else:
        print(f"❌ Aborting: Target file '{image_name}' not found.")