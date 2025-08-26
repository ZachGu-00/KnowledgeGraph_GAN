#!/usr/bin/env python3
"""
Train improved discriminator with hard negatives and optimized architecture
python train_discriminator_v2.py --device cuda --batch_size 32 --lr 5e-5 --epochs 2 --hop 1hop
"""
import torch
import torch.optim as optim
from data_loader import create_data_loaders
from embeddings import OptimizedEmbeddingManager as EmbeddingManager
from models.discriminator_v2 import ImprovedMultiHeadDiscriminator, HardNegativeSampler, ImprovedGraphletBuilder
import argparse
import time
from pathlib import Path
import pickle

class ImprovedDiscriminatorTrainer:
    """Trainer with hard negative sampling and adaptive loss weights"""
    
    def __init__(self, discriminator, dataset, device='cuda', embedding_manager=None):
        self.discriminator = discriminator
        self.dataset = dataset
        self.device = device
        self.embedding_manager = embedding_manager
        
        # Load knowledge graph
        with open('graph/knowledge_graph.pkl', 'rb') as f:
            self.graph = pickle.load(f)
        
        # Initialize hard negative sampler
        self.hard_negative_sampler = HardNegativeSampler(
            self.graph, 
            self.embedding_manager.entity_dict,
            self.embedding_manager
        )
        
        # Initialize improved graphlet builder
        self.graphlet_builder = ImprovedGraphletBuilder(dataset, embedding_manager)
        
        self.discriminator.to(device)
        
        # Loss function
        self.margin = 2.0  # Increased margin for harder separation
        self.criterion = torch.nn.MarginRankingLoss(margin=self.margin)
        
        # Adaptive head weights for 1-hop (less structural complexity)
        self.head_weights = {
            'structural': 0.1,   # Reduced for 1-hop
            'semantic': 0.5,     # Increased importance
            'logic': 0.4         # Increased importance
        }
        
        # Statistics tracking
        self.stats = {
            'total_batches': 0,
            'successful_batches': 0,
            'hard_negatives_generated': 0,
            'total_samples': 0,
            'loss_history': []
        }
    
    def create_training_samples(self, batch):
        """Create samples with hard negatives"""
        positive_samples = []
        negative_samples = []
        negative_types = []
        questions = []
        
        for i, golden_paths in enumerate(batch['golden_paths']):
            if not golden_paths:
                continue
            
            question = batch['questions'][i]
            question_emb = self.embedding_manager.encode_question(question).clone()
            
            # Use first golden path
            golden_path = golden_paths[0]
            
            try:
                # Build positive graphlet with improved features
                positive_graphlet = self.graphlet_builder.build_evidence_graphlet(golden_path)
                positive_samples.append(positive_graphlet)
                questions.append(question_emb)
                
                # Generate hard negatives
                hard_negatives = self.hard_negative_sampler.generate_hard_negatives(
                    golden_path, num_negatives=2
                )
                
                for neg_path in hard_negatives:
                    try:
                        negative_graphlet = self.graphlet_builder.build_evidence_graphlet(neg_path)
                        negative_samples.append(negative_graphlet)
                        negative_types.append(neg_path['type'])
                    except Exception as e:
                        continue
                
                self.stats['hard_negatives_generated'] += len(hard_negatives)
                
            except Exception as e:
                print(f"Error building graphlet: {e}")
                continue
        
        return positive_samples, negative_samples, questions, negative_types
    
    def compute_loss(self, pos_scores, neg_scores):
        """Compute multi-head loss with adaptive weights"""
        total_loss = 0
        losses = {}
        
        for head in ['structural', 'semantic', 'logic']:
            pos_head_scores = pos_scores[head]
            neg_head_scores = neg_scores[head]
            
            # Ensure matching dimensions
            num_neg_per_pos = len(neg_head_scores) // len(pos_head_scores)
            if num_neg_per_pos > 0:
                expanded_pos = pos_head_scores.repeat(num_neg_per_pos, 1)
            else:
                expanded_pos = pos_head_scores
                neg_head_scores = neg_head_scores[:len(pos_head_scores)]
            
            # Target: positive should rank higher
            target = torch.ones(len(expanded_pos)).to(self.device)
            
            # Margin ranking loss
            loss = self.criterion(expanded_pos.squeeze(), neg_head_scores.squeeze(), target)
            losses[head] = loss
            total_loss += self.head_weights[head] * loss
        
        return total_loss, losses
    
    def train_epoch(self, dataloader, optimizer, epoch=0):
        """Train for one epoch with monitoring"""
        self.discriminator.train()
        total_loss = 0
        num_batches = 0
        
        print(f"Starting epoch {epoch + 1} with hard negative sampling...")
        
        for batch_idx, batch in enumerate(dataloader):
            if batch is None:
                continue
            
            optimizer.zero_grad()
            
            try:
                # Create samples with hard negatives
                pos_samples, neg_samples, questions, neg_types = self.create_training_samples(batch)
                
                if not pos_samples or not neg_samples:
                    continue
                
                if batch_idx == 0:
                    print(f"  Batch 0: {len(pos_samples)} positive, {len(neg_samples)} negative samples")
                    type_counts = {}
                    for t in neg_types:
                        type_counts[t] = type_counts.get(t, 0) + 1
                    print(f"  Negative types: {type_counts}")
                
                # Forward pass for positive samples
                pos_scores_list = []
                for i, pos_sample in enumerate(pos_samples):
                    for key in pos_sample:
                        if isinstance(pos_sample[key], torch.Tensor):
                            pos_sample[key] = pos_sample[key].to(self.device)
                    
                    question_emb = questions[i].to(self.device) if i < len(questions) else None
                    pos_scores = self.discriminator(pos_sample, question_emb)
                    pos_scores_list.append(pos_scores)
                
                # Forward pass for negative samples
                neg_scores_list = []
                for neg_sample in neg_samples:
                    for key in neg_sample:
                        if isinstance(neg_sample[key], torch.Tensor):
                            neg_sample[key] = neg_sample[key].to(self.device)
                    
                    # Use same question for negatives
                    neg_scores = self.discriminator(neg_sample, questions[0].to(self.device) if questions else None)
                    neg_scores_list.append(neg_scores)
                
                if not pos_scores_list or not neg_scores_list:
                    continue
                
                # Combine scores
                pos_scores_combined = {}
                neg_scores_combined = {}
                
                for head in ['structural', 'semantic', 'logic']:
                    pos_scores_combined[head] = torch.cat([s[head] for s in pos_scores_list], dim=0)
                    neg_scores_combined[head] = torch.cat([s[head] for s in neg_scores_list], dim=0)
                
                # Compute loss
                loss, head_losses = self.compute_loss(pos_scores_combined, neg_scores_combined)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                self.stats['loss_history'].append(loss.item())
                
                if batch_idx % 500 == 0 and batch_idx > 0:
                    avg_loss = total_loss / num_batches
                    print(f"  Batch {batch_idx}: loss={avg_loss:.4f}")
                    for head, head_loss in head_losses.items():
                        print(f"    {head}: {head_loss.item():.4f}")
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        # Print epoch statistics
        print(f"\nEpoch {epoch + 1} Statistics:")
        print(f"  Average loss: {avg_loss:.4f}")
        print(f"  Total hard negatives generated: {self.stats['hard_negatives_generated']}")
        print(f"  Successful batches: {num_batches}")
        
        return avg_loss

def main():
    parser = argparse.ArgumentParser(description='Train Improved Discriminator')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--hop', type=str, default='1hop', help='Hop type')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    
    args = parser.parse_args()
    
    print("=== Improved Discriminator Training ===")
    print(f"Device: {args.device}, Batch size: {args.batch_size}, LR: {args.lr}, Epochs: {args.epochs}")
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    
    # Load SBERT embeddings
    print("\nLoading SBERT embeddings...")
    embedding_manager = EmbeddingManager(device=args.device)
    
    # Load data
    datasets, loaders = create_data_loaders(
        hop=args.hop,
        batch_size=args.batch_size
    )
    
    train_dataset = datasets['train']
    train_loader = loaders['train']
    
    print(f"Training samples: {len(train_dataset)}")
    
    # Create improved discriminator
    print("\nInitializing improved discriminator...")
    discriminator = ImprovedMultiHeadDiscriminator(
        embedding_dim=embedding_manager.embedding_dim,
        hidden_dim=args.hidden_dim
    )
    
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    discriminator.to(device)
    optimizer = optim.AdamW(discriminator.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Create trainer
    trainer = ImprovedDiscriminatorTrainer(
        discriminator, train_dataset, device=args.device, embedding_manager=embedding_manager
    )
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        # Reset epoch stats
        trainer.stats['hard_negatives_generated'] = 0
        
        # Train
        loss = trainer.train_epoch(train_loader, optimizer, epoch)
        
        # Step scheduler
        scheduler.step()
        
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
        
        # Save best model
        if loss < best_loss:
            best_loss = loss
            print(f"New best loss: {best_loss:.4f}, saving model...")
            
            save_dir = Path("checkpoints")
            save_dir.mkdir(exist_ok=True)
            
            torch.save({
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'embedding_dim': embedding_manager.embedding_dim,
                'hidden_dim': args.hidden_dim,
                'epoch': epoch + 1,
                'loss': loss,
                'args': args
            }, save_dir / f"discriminator_v2_{args.hop}.pth")
        
        if args.device == 'cuda':
            print(f"GPU Memory: {torch.cuda.memory_allocated(device) / 1024**2:.1f} MB")
    
    print(f"\nTraining complete! Best loss: {best_loss:.4f}")
    print(f"Model saved to checkpoints/discriminator_v2_{args.hop}.pth")

if __name__ == "__main__":
    main()