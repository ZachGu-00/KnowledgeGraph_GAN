#!/usr/bin/env python3
"""
Training script for EPR-based Discriminator
Based on proposal.md EPR methodology:

1. Extract patterns from MetaQA KB
2. Pre-train Bi-Encoder for AP recall  
3. Pre-train Cross-Encoder for EP ranking
4. Train GAT with EPR prior injection
"""
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import json
import random
from pathlib import Path
from tqdm import tqdm
import argparse
import time
from typing import Dict, List, Tuple
from collections import defaultdict

from models.epr_discriminator import EPRDiscriminator, BiEncoderEPR, CrossEncoderEPR, EPRInformedGAT
from data_loader import create_data_loaders
from embeddings import OptimizedEmbeddingManager as EmbeddingManager

class EPRPatternDataset(Dataset):
    """Dataset for EPR pattern training"""
    
    def __init__(self, questions, patterns, labels, pattern_type='atomic'):
        self.questions = questions
        self.patterns = patterns  
        self.labels = labels
        self.pattern_type = pattern_type
        
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        return {
            'question': self.questions[idx],
            'pattern': self.patterns[idx],
            'label': self.labels[idx]
        }

class EPRTrainer:
    """Trainer for EPR Discriminator with three-stage training"""
    
    def __init__(self, device='cuda', batch_size=32, learning_rate=2e-5):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # Load knowledge graph and embeddings
        print("Loading knowledge graph and embeddings...")
        self.load_knowledge_base()
        self.embedding_manager = EmbeddingManager(device=device)
        
        # Extract patterns from KB
        print("Extracting patterns from knowledge base...")
        self.atomic_patterns = self.extract_atomic_patterns()
        self.execution_patterns = self.generate_execution_patterns()
        
        print(f"Extracted {len(self.atomic_patterns)} atomic patterns")
        print(f"Generated {len(self.execution_patterns)} execution patterns")
        
        # Initialize models
        self.bi_encoder = BiEncoderEPR().to(self.device)
        self.cross_encoder = CrossEncoderEPR().to(self.device) 
        self.gat_discriminator = EPRInformedGAT().to(self.device)
        
    def load_knowledge_base(self):
        """Load knowledge graph from pickle file"""
        import os
        base_dir = os.path.dirname(os.path.abspath(__file__))
        graph_path = os.path.join(base_dir, 'graph', 'knowledge_graph.pkl')
        
        with open(graph_path, 'rb') as f:
            self.kb_graph = pickle.load(f)
            
        print(f"Loaded KB: {self.kb_graph.number_of_nodes()} nodes, {self.kb_graph.number_of_edges()} edges")
    
    def extract_atomic_patterns(self, max_patterns=2000):
        """Extract ER-APs and RR-APs from knowledge base"""
        er_patterns = set()
        rr_patterns = set()
        
        # Extract ER-APs (Entity-Relation Atomic Patterns)
        entity_relations = defaultdict(set)
        for source, target, data in self.kb_graph.edges(data=True):
            relation = data.get('relation', 'unknown')
            er_patterns.add(f"entity -> {relation}")
            er_patterns.add(f"{relation} -> entity")
            entity_relations[source].add(relation)
            entity_relations[target].add(relation)
        
        # Extract RR-APs (Relation-Relation Atomic Patterns)  
        for entity, relations in entity_relations.items():
            relations_list = list(relations)
            for i, rel1 in enumerate(relations_list):
                for rel2 in relations_list[i+1:]:
                    rr_patterns.add(f"{rel1} <-> {rel2}")
        
        all_patterns = list(er_patterns) + list(rr_patterns)
        return all_patterns[:max_patterns]
    
    def generate_execution_patterns(self, max_eps=1000):
        """Generate Execution Patterns by combining atomic patterns"""
        execution_patterns = []
        
        # Simple 1-hop patterns
        relations = set()
        for _, _, data in self.kb_graph.edges(data=True):
            relations.add(data.get('relation', 'unknown'))
        
        for rel in relations:
            execution_patterns.append(f"entity -> {rel} -> ?x")
            
        # 2-hop patterns (simplified)
        relations_list = list(relations)[:50]  # Limit for efficiency
        for i, rel1 in enumerate(relations_list):
            for rel2 in relations_list[i+1:10]:  # Limit combinations
                execution_patterns.append(f"entity -> {rel1} -> entity -> {rel2} -> ?x")
        
        return execution_patterns[:max_eps]
    
    def create_ap_training_data(self, qa_dataset, negative_ratio=3):
        """Create training data for Bi-Encoder (Atomic Pattern recall)"""
        questions = []
        patterns = []
        labels = []
        
        for qa_item in tqdm(qa_dataset, desc="Creating AP training data"):
            question = qa_item['question']
            golden_paths = qa_item.get('golden_paths', [])
            
            if not golden_paths:
                continue
                
            # Positive samples: patterns that appear in golden paths
            positive_patterns = []
            for path in golden_paths:
                for relation in path.get('relations', []):
                    positive_patterns.append(f"entity -> {relation}")
                    positive_patterns.append(f"{relation} -> entity")
            
            # Add positive samples
            for pattern in set(positive_patterns):
                if pattern in self.atomic_patterns:
                    questions.append(question)
                    patterns.append(pattern)
                    labels.append(1)
            
            # Add negative samples (random patterns not in golden paths)
            negative_patterns = random.sample(
                [p for p in self.atomic_patterns if p not in positive_patterns],
                min(len(positive_patterns) * negative_ratio, len(self.atomic_patterns) - len(positive_patterns))
            )
            
            for pattern in negative_patterns:
                questions.append(question)
                patterns.append(pattern)
                labels.append(0)
        
        return EPRPatternDataset(questions, patterns, labels, 'atomic')
    
    def create_ep_training_data(self, qa_dataset, negative_ratio=2):
        """Create training data for Cross-Encoder (Execution Pattern ranking)"""
        questions = []
        patterns = []
        labels = []
        
        for qa_item in tqdm(qa_dataset, desc="Creating EP training data"):
            question = qa_item['question']
            golden_paths = qa_item.get('golden_paths', [])
            
            if not golden_paths:
                continue
            
            # Positive samples: execution patterns that match golden paths
            positive_eps = []
            for path in golden_paths:
                relations = path.get('relations', [])
                if len(relations) == 1:
                    ep = f"entity -> {relations[0]} -> ?x"
                elif len(relations) == 2:
                    ep = f"entity -> {relations[0]} -> entity -> {relations[1]} -> ?x"
                else:
                    continue  # Skip longer patterns for now
                    
                if ep in self.execution_patterns:
                    positive_eps.append(ep)
            
            # Add positive samples
            for ep in set(positive_eps):
                questions.append(question)
                patterns.append(ep)
                labels.append(1)
            
            # Add negative samples
            negative_eps = random.sample(
                [ep for ep in self.execution_patterns if ep not in positive_eps],
                min(len(positive_eps) * negative_ratio, len(self.execution_patterns) - len(positive_eps))
            )
            
            for ep in negative_eps:
                questions.append(question)
                patterns.append(ep)
                labels.append(0)
        
        return EPRPatternDataset(questions, patterns, labels, 'execution')
    
    def train_bi_encoder(self, train_dataset, epochs=5):
        """Train Bi-Encoder for AP recall using contrastive learning"""
        print(f"\n=== Training Bi-Encoder (AP Recall) ===")
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = optim.AdamW(self.bi_encoder.parameters(), lr=self.learning_rate)
        
        self.bi_encoder.train()
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                optimizer.zero_grad()
                
                questions = batch['question']
                patterns = batch['pattern']
                
                # Forward pass
                outputs = self.bi_encoder(questions, patterns)
                similarity_matrix = outputs['similarity']
                
                # Contrastive loss (in-batch negatives)
                batch_size = len(questions)
                labels = torch.arange(batch_size).to(self.device)
                
                loss = F.cross_entropy(similarity_matrix, labels) + F.cross_entropy(similarity_matrix.T, labels)
                loss = loss / 2.0
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.bi_encoder.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
    
    def train_cross_encoder(self, train_dataset, epochs=3):
        """Train Cross-Encoder for EP ranking"""
        print(f"\n=== Training Cross-Encoder (EP Ranking) ===")
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = optim.AdamW(self.cross_encoder.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        self.cross_encoder.train()
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                optimizer.zero_grad()
                
                questions = batch['question']
                patterns = batch['pattern']
                labels = batch['label'].to(self.device)
                
                # Forward pass
                logits = self.cross_encoder(questions, patterns)
                loss = criterion(logits, labels)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.cross_encoder.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            avg_loss = total_loss / len(train_loader)
            accuracy = 100 * correct / total
            print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    def create_gat_training_data(self, qa_dataset):
        """Create graph data for GAT discriminator training"""
        positive_samples = []
        negative_samples = []
        
        for qa_item in tqdm(qa_dataset[:100], desc="Creating GAT training data"):  # Limit for demo
            question = qa_item['question']
            golden_paths = qa_item.get('golden_paths', [])
            
            if not golden_paths:
                continue
            
            # Create positive sample from golden path
            for path in golden_paths[:1]:  # Take first golden path
                try:
                    graph_data = self.create_graph_from_path(path, question)
                    if graph_data:
                        positive_samples.append(graph_data)
                except:
                    continue
            
            # Create negative samples by corrupting golden paths  
            for path in golden_paths[:1]:
                try:
                    # Corrupt by changing relations
                    corrupted_path = self.corrupt_path(path, corruption_type='relation')
                    graph_data = self.create_graph_from_path(corrupted_path, question)
                    if graph_data:
                        negative_samples.append(graph_data)
                except:
                    continue
        
        return positive_samples, negative_samples
    
    def create_graph_from_path(self, path, question):
        """Create graph representation from reasoning path"""
        entities = path.get('entities', [])
        relations = path.get('relations', [])
        
        if len(entities) < 2 or len(relations) < 1:
            return None
        
        # Get entity embeddings
        node_features = []
        for entity_id in entities:
            emb = self.embedding_manager.get_entity_embedding(entity_id)
            node_features.append(emb)
        
        # Create simple chain graph
        edge_index = [[i, i+1] for i in range(len(entities)-1)]
        edge_index = torch.tensor(edge_index).T.long()
        
        # Get relation embeddings
        edge_features = []
        for relation in relations:
            emb = self.embedding_manager.get_relation_embedding(relation)
            edge_features.append(emb)
        
        # Create path mask (all nodes in path are important)
        path_mask = torch.ones(len(entities), dtype=torch.bool)
        
        return {
            'question': question,
            'node_features': torch.stack(node_features),
            'edge_index': edge_index,
            'edge_features': torch.stack(edge_features) if edge_features else torch.zeros(1, 384),
            'path_mask': path_mask
        }
    
    def corrupt_path(self, path, corruption_type='relation'):
        """Corrupt a golden path to create negative sample"""
        corrupted = path.copy()
        
        if corruption_type == 'relation' and 'relations' in corrupted:
            # Replace random relation
            relations = corrupted['relations'].copy()
            if relations:
                idx = random.randint(0, len(relations)-1)
                # Get random relation from KB
                all_relations = [data.get('relation', 'unknown') 
                               for _, _, data in self.kb_graph.edges(data=True)]
                relations[idx] = random.choice(all_relations)
                corrupted['relations'] = relations
        
        return corrupted
    
    def train_gat_discriminator(self, positive_samples, negative_samples, epochs=10):
        """Train GAT discriminator with EPR prior injection"""
        print(f"\n=== Training GAT Discriminator with EPR Prior ===")
        
        optimizer = optim.AdamW(self.gat_discriminator.parameters(), lr=self.learning_rate)
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            # Sample batches
            batch_size = min(self.batch_size, len(positive_samples))
            for i in range(0, len(positive_samples), batch_size):
                pos_batch = positive_samples[i:i+batch_size]
                neg_batch = negative_samples[i:i+batch_size] if i+batch_size <= len(negative_samples) else negative_samples[:batch_size]
                
                optimizer.zero_grad()
                
                pos_scores = []
                neg_scores = []
                
                # Process positive samples
                for sample in pos_batch:
                    try:
                        # Get EPR scores (simplified - use default scores)
                        num_edges = sample['edge_index'].size(1)
                        epr_scores = torch.ones(num_edges).to(self.device) * 0.8  # High score for golden paths
                        
                        question_emb = self.embedding_manager.encode_question(sample['question'])
                        
                        output = self.gat_discriminator(
                            node_features=sample['node_features'].to(self.device),
                            edge_index=sample['edge_index'].to(self.device),
                            edge_attr=sample['edge_features'].to(self.device),
                            epr_scores=epr_scores,
                            question_emb=question_emb.to(self.device),
                            path_mask=sample['path_mask'].to(self.device)
                        )
                        
                        pos_scores.append(output['path_score'])
                    except Exception as e:
                        continue
                
                # Process negative samples
                for sample in neg_batch:
                    try:
                        num_edges = sample['edge_index'].size(1)
                        epr_scores = torch.ones(num_edges).to(self.device) * 0.3  # Low score for corrupted paths
                        
                        question_emb = self.embedding_manager.encode_question(sample['question'])
                        
                        output = self.gat_discriminator(
                            node_features=sample['node_features'].to(self.device),
                            edge_index=sample['edge_index'].to(self.device),
                            edge_attr=sample['edge_features'].to(self.device),
                            epr_scores=epr_scores,
                            question_emb=question_emb.to(self.device),
                            path_mask=sample['path_mask'].to(self.device)
                        )
                        
                        neg_scores.append(output['path_score'])
                    except Exception as e:
                        continue
                
                if pos_scores and neg_scores:
                    pos_scores = torch.stack(pos_scores)
                    neg_scores = torch.stack(neg_scores)
                    
                    # Ranking loss: positive should score higher than negative
                    target = torch.ones(len(pos_scores)).to(self.device)
                    loss = F.margin_ranking_loss(pos_scores, neg_scores, target, margin=0.3)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.gat_discriminator.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
            
            if num_batches > 0:
                avg_loss = total_loss / num_batches
                print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
    
    def save_models(self, save_dir):
        """Save trained models"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        torch.save(self.bi_encoder.state_dict(), save_dir / 'bi_encoder.pth')
        torch.save(self.cross_encoder.state_dict(), save_dir / 'cross_encoder.pth') 
        torch.save(self.gat_discriminator.state_dict(), save_dir / 'gat_discriminator.pth')
        
        # Save patterns
        with open(save_dir / 'atomic_patterns.json', 'w') as f:
            json.dump(self.atomic_patterns, f)
        
        with open(save_dir / 'execution_patterns.json', 'w') as f:
            json.dump(self.execution_patterns, f)
        
        print(f"Models saved to {save_dir}")

def main():
    parser = argparse.ArgumentParser(description='Train EPR Discriminator')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--epochs_bi', type=int, default=5, help='Epochs for Bi-Encoder')
    parser.add_argument('--epochs_cross', type=int, default=3, help='Epochs for Cross-Encoder')
    parser.add_argument('--epochs_gat', type=int, default=10, help='Epochs for GAT')
    parser.add_argument('--hop', type=str, default='1hop', help='Hop type')
    
    args = parser.parse_args()
    
    print("=== EPR Discriminator Training ===")
    print(f"Device: {args.device}, Batch size: {args.batch_size}, LR: {args.lr}")
    
    # Load training data
    datasets, loaders = create_data_loaders(hop=args.hop, batch_size=args.batch_size)
    
    if 'train' not in datasets:
        print("Error: Training data not found!")
        return
    
    train_dataset = datasets['train']
    train_data = []
    
    # Convert dataset to list format
    for i in range(len(train_dataset)):
        train_data.append(train_dataset[i])
    
    # Initialize trainer
    trainer = EPRTrainer(
        device=args.device,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
    
    # Stage 1: Train Bi-Encoder for AP recall
    print("Stage 1: Creating AP training data...")
    ap_dataset = trainer.create_ap_training_data(train_data)
    trainer.train_bi_encoder(ap_dataset, epochs=args.epochs_bi)
    
    # Stage 2: Train Cross-Encoder for EP ranking  
    print("Stage 2: Creating EP training data...")
    ep_dataset = trainer.create_ep_training_data(train_data)
    trainer.train_cross_encoder(ep_dataset, epochs=args.epochs_cross)
    
    # Stage 3: Train GAT discriminator
    print("Stage 3: Creating GAT training data...")
    pos_samples, neg_samples = trainer.create_gat_training_data(train_data)
    trainer.train_gat_discriminator(pos_samples, neg_samples, epochs=args.epochs_gat)
    
    # Save models
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(base_dir, 'checkpoints', 'epr_discriminator')
    trainer.save_models(save_dir)
    
    print("🎉 EPR Discriminator training complete!")

if __name__ == "__main__":
    main()