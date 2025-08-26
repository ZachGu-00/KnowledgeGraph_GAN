#!/usr/bin/env python3
"""
Test the trained discriminator on 1-hop test dataset
"""
import torch
import torch.nn.functional as F
from pathlib import Path
import random
import numpy as np
from data_loader import create_data_loaders
from embeddings import OptimizedEmbeddingManager as EmbeddingManager
from models.discriminator import MultiHeadDiscriminator, GraphletBuilder
import pickle

class DiscriminatorTester:
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load embeddings
        print("Loading SBERT embeddings...")
        self.embedding_manager = EmbeddingManager(device=self.device)
        
        # Load discriminator
        print(f"Loading discriminator from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.discriminator = MultiHeadDiscriminator(
            embedding_dim=checkpoint['embedding_dim'],
            hidden_dim=checkpoint['hidden_dim']
        )
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.discriminator.to(self.device)
        self.discriminator.eval()
        
        print(f"Discriminator loaded: {sum(p.numel() for p in self.discriminator.parameters()):,} parameters")
        
        # Load knowledge graph
        with open('graph/knowledge_graph.pkl', 'rb') as f:
            self.graph = pickle.load(f)
    
    def create_negative_samples(self, golden_path, dataset):
        """Create different types of negative samples"""
        negative_samples = []
        head_id = golden_path['head_id']
        correct_relation = golden_path['relations'][0]
        correct_answer_id = golden_path['answer_id']
        
        # Type 1: Wrong relation
        wrong_relations = [r for r in self.embedding_manager.relations if r != correct_relation]
        if wrong_relations and self.graph.has_node(head_id):
            wrong_rel = random.choice(wrong_relations)
            for _, target, edge_data in self.graph.edges(head_id, data=True):
                if edge_data.get('relation') == wrong_rel:
                    negative_samples.append({
                        'head_id': head_id,
                        'entities': [head_id, target],
                        'relations': [wrong_rel],
                        'answer_id': target,
                        'answer_name': self.embedding_manager.entity_dict.get(target, 'unknown'),
                        'type': 'wrong_relation'
                    })
                    break
        
        # Type 2: Wrong target with correct relation
        if self.graph.has_node(head_id):
            same_relation_targets = []
            for _, target, edge_data in self.graph.edges(head_id, data=True):
                if edge_data.get('relation') == correct_relation and target != correct_answer_id:
                    same_relation_targets.append(target)
            
            if same_relation_targets:
                wrong_target = random.choice(same_relation_targets)
                negative_samples.append({
                    'head_id': head_id,
                    'entities': [head_id, wrong_target],
                    'relations': [correct_relation],
                    'answer_id': wrong_target,
                    'answer_name': self.embedding_manager.entity_dict.get(wrong_target, 'unknown'),
                    'type': 'wrong_target'
                })
        
        # Type 3: Random entity
        if not negative_samples:
            all_entities = list(self.embedding_manager.entity_dict.keys())
            random_entity = random.choice([e for e in all_entities 
                                          if e != head_id and e != correct_answer_id])
            negative_samples.append({
                'head_id': head_id,
                'entities': [head_id, random_entity],
                'relations': [correct_relation],
                'answer_id': random_entity,
                'answer_name': self.embedding_manager.entity_dict.get(random_entity, 'unknown'),
                'type': 'random'
            })
        
        return negative_samples
    
    def evaluate_ranking(self, test_loader):
        """Evaluate discriminator's ranking ability"""
        print("\n=== Evaluating Discriminator Ranking ===")
        
        graphlet_builder = GraphletBuilder(test_loader.dataset, self.embedding_manager)
        
        results = {
            'total': 0,
            'correct_rankings': 0,
            'structural_correct': 0,
            'semantic_correct': 0,
            'logic_correct': 0,
            'head_scores': {'structural': [], 'semantic': [], 'logic': []},
            'score_differences': []
        }
        
        sample_count = 0
        total_test_samples = len(test_loader.dataset)
        max_samples = int(total_test_samples * 0.2)  # Use 20% of test data
        print(f"Testing on {max_samples} samples (20% of {total_test_samples})")
        
        print(f"Starting evaluation...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if batch is None:
                    print(f"Batch {batch_idx}: None")
                    continue
                    
                if sample_count >= max_samples:
                    break
                
                if batch_idx == 0:
                    print(f"First batch size: {len(batch['questions'])}")
                
                for i, golden_paths in enumerate(batch['golden_paths']):
                    if sample_count >= max_samples:
                        break
                        
                    if not golden_paths:
                        if batch_idx == 0 and i < 3:
                            print(f"  Sample {i}: No golden paths")
                        continue
                    
                    golden_path = golden_paths[0]
                    if batch_idx == 0 and i == 0:
                        print(f"  First golden path: {golden_path}")
                    question = batch['questions'][i]
                    question_emb = self.embedding_manager.encode_question(question).to(self.device)
                    
                    # Build positive graphlet
                    try:
                        pos_graphlet = graphlet_builder.build_evidence_graphlet(golden_path)
                        for key in pos_graphlet:
                            if isinstance(pos_graphlet[key], torch.Tensor):
                                pos_graphlet[key] = pos_graphlet[key].to(self.device)
                        
                        pos_scores = self.discriminator(pos_graphlet, question_emb)
                        
                        # Generate and evaluate negative samples
                        neg_samples = self.create_negative_samples(golden_path, test_loader.dataset)
                        
                        for neg_path in neg_samples[:1]:  # Test with 1 negative per positive
                            try:
                                neg_graphlet = graphlet_builder.build_evidence_graphlet(neg_path)
                                for key in neg_graphlet:
                                    if isinstance(neg_graphlet[key], torch.Tensor):
                                        neg_graphlet[key] = neg_graphlet[key].to(self.device)
                                
                                neg_scores = self.discriminator(neg_graphlet, question_emb)
                                
                                # Evaluate ranking for each head
                                results['total'] += 1
                                
                                # Overall ranking
                                if pos_scores['total'].item() > neg_scores['total'].item():
                                    results['correct_rankings'] += 1
                                
                                # Per-head ranking
                                if pos_scores['structural'].item() > neg_scores['structural'].item():
                                    results['structural_correct'] += 1
                                
                                if pos_scores['semantic'].item() > neg_scores['semantic'].item():
                                    results['semantic_correct'] += 1
                                
                                if pos_scores['logic'].item() > neg_scores['logic'].item():
                                    results['logic_correct'] += 1
                                
                                # Record score differences
                                score_diff = pos_scores['total'].item() - neg_scores['total'].item()
                                results['score_differences'].append(score_diff)
                                
                                # Record individual head scores
                                for head in ['structural', 'semantic', 'logic']:
                                    results['head_scores'][head].append({
                                        'pos': pos_scores[head].item(),
                                        'neg': neg_scores[head].item(),
                                        'diff': pos_scores[head].item() - neg_scores[head].item()
                                    })
                                
                                sample_count += 1
                                
                                if sample_count % 200 == 0 and sample_count > 0:
                                    accuracy_so_far = results['correct_rankings'] / results['total']
                                    print(f"Processed {sample_count}/{max_samples} samples... Current accuracy: {accuracy_so_far:.3f}")
                                
                            except Exception as e:
                                continue
                    
                    except Exception as e:
                        continue
        
        return results
    
    def print_results(self, results):
        """Print evaluation results"""
        print("\n=== Discriminator Test Results ===")
        print(f"Total test samples: {results['total']}")
        
        if results['total'] > 0:
            # Overall accuracy
            overall_acc = results['correct_rankings'] / results['total']
            print(f"\nOverall Ranking Accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)")
            
            # Per-head accuracy
            print("\nPer-Head Ranking Accuracy:")
            structural_acc = results['structural_correct'] / results['total']
            semantic_acc = results['semantic_correct'] / results['total']
            logic_acc = results['logic_correct'] / results['total']
            
            print(f"  Structural Head: {structural_acc:.4f} ({structural_acc*100:.2f}%)")
            print(f"  Semantic Head: {semantic_acc:.4f} ({semantic_acc*100:.2f}%)")
            print(f"  Logic Head: {logic_acc:.4f} ({logic_acc*100:.2f}%)")
            
            # Score statistics
            score_diffs = results['score_differences']
            print(f"\nScore Differences (Positive - Negative):")
            print(f"  Mean: {np.mean(score_diffs):.4f}")
            print(f"  Std: {np.std(score_diffs):.4f}")
            print(f"  Min: {np.min(score_diffs):.4f}")
            print(f"  Max: {np.max(score_diffs):.4f}")
            
            # Percentage with positive difference
            positive_diff = sum(1 for d in score_diffs if d > 0) / len(score_diffs)
            print(f"  Positive difference rate: {positive_diff:.4f} ({positive_diff*100:.2f}%)")
            
            # Per-head score analysis
            print("\nPer-Head Score Analysis:")
            for head in ['structural', 'semantic', 'logic']:
                head_data = results['head_scores'][head]
                pos_scores = [d['pos'] for d in head_data]
                neg_scores = [d['neg'] for d in head_data]
                diffs = [d['diff'] for d in head_data]
                
                print(f"\n  {head.capitalize()} Head:")
                print(f"    Positive scores: mean={np.mean(pos_scores):.4f}, std={np.std(pos_scores):.4f}")
                print(f"    Negative scores: mean={np.mean(neg_scores):.4f}, std={np.std(neg_scores):.4f}")
                print(f"    Score difference: mean={np.mean(diffs):.4f}, std={np.std(diffs):.4f}")
                print(f"    Positive diff rate: {sum(1 for d in diffs if d > 0)/len(diffs):.4f}")

def main():
    # Set random seed
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Check if checkpoint exists
    checkpoint_path = "checkpoints/discriminator_solo_trained_1hop.pth"
    if not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    # Load test data
    print("Loading test dataset...")
    datasets, loaders = create_data_loaders(hop="1hop", batch_size=32)
    test_loader = loaders['test']
    
    # Create tester
    tester = DiscriminatorTester(checkpoint_path, device='cuda')
    
    # Run evaluation
    results = tester.evaluate_ranking(test_loader)
    
    # Print results
    tester.print_results(results)
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    main()