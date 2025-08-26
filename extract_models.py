"""
从检查点中提取生成器和判别器模型并分开保存
"""
import torch
from models.generator import PathGenerator
from models.discriminator import MultiHeadDiscriminator
from pathlib import Path

def extract_models_from_checkpoint(checkpoint_path):
    """从检查点提取并分开保存生成器和判别器"""
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # 获取模型参数
    embedding_dim = checkpoint['embedding_dims']['entity']  # 384 SBERT
    num_relations = checkpoint['num_relations']
    args = checkpoint['args']
    
    print(f"Checkpoint info:")
    print(f"  Embedding dim: {embedding_dim}")
    print(f"  Number of relations: {num_relations}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Relations: {checkpoint['relations']}")
    
    # 创建生成器
    print("\n1. Extracting Generator...")
    generator = PathGenerator(
        embedding_dim=embedding_dim,
        num_relations=num_relations,
        hidden_dim=args.hidden_dim
    )
    generator.load_state_dict(checkpoint['generator_state_dict'])
    
    # 保存生成器
    generator_save_path = "checkpoints/generator_1hop.pth"
    torch.save({
        'model_state_dict': generator.state_dict(),
        'embedding_dim': embedding_dim,
        'num_relations': num_relations,
        'relations': checkpoint['relations'],
        'hidden_dim': args.hidden_dim,
        'model_type': 'PathGenerator'
    }, generator_save_path)
    print(f"OK Generator saved to: {generator_save_path}")
    
    # 创建判别器
    print("\n2. Extracting Discriminator...")
    discriminator = MultiHeadDiscriminator(
        embedding_dim=embedding_dim,
        hidden_dim=args.hidden_dim
    )
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    
    # 保存判别器
    discriminator_save_path = "checkpoints/discriminator_1hop.pth"
    torch.save({
        'model_state_dict': discriminator.state_dict(),
        'embedding_dim': embedding_dim,
        'hidden_dim': args.hidden_dim,
        'model_type': 'MultiHeadDiscriminator'
    }, discriminator_save_path)
    print(f"OK Discriminator saved to: {discriminator_save_path}")
    
    return generator, discriminator

def load_generator_only(generator_path, device='cpu'):
    """只加载生成器"""
    checkpoint = torch.load(generator_path, map_location=device, weights_only=False)
    
    generator = PathGenerator(
        embedding_dim=checkpoint['embedding_dim'],
        num_relations=checkpoint['num_relations'],
        hidden_dim=checkpoint['hidden_dim']
    )
    generator.load_state_dict(checkpoint['model_state_dict'])
    generator.to(device)
    generator.eval()
    
    return generator, checkpoint

def load_discriminator_only(discriminator_path, device='cpu'):
    """只加载判别器"""
    checkpoint = torch.load(discriminator_path, map_location=device, weights_only=False)
    
    discriminator = MultiHeadDiscriminator(
        embedding_dim=checkpoint['embedding_dim'],
        hidden_dim=checkpoint['hidden_dim']
    )
    discriminator.load_state_dict(checkpoint['model_state_dict'])
    discriminator.to(device)
    discriminator.eval()
    
    return discriminator, checkpoint

if __name__ == "__main__":
    # 检查是否存在检查点
    checkpoint_path = "checkpoints/pretrained_1hop.pth"
    if not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        exit(1)
    
    # 提取模型
    generator, discriminator = extract_models_from_checkpoint(checkpoint_path)
    
    print(f"\n3. Model Summary:")
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # 测试加载
    print("\n4. Testing individual loading...")
    gen, gen_info = load_generator_only("checkpoints/generator_1hop.pth")
    disc, disc_info = load_discriminator_only("checkpoints/discriminator_1hop.pth")
    
    print("OK Individual model loading works!")
    print("\nExtraction complete!")