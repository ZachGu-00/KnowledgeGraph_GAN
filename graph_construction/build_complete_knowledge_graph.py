import numpy as np
import networkx as nx
import pickle
import json
from pathlib import Path
from collections import defaultdict, Counter
import os

class CompleteKnowledgeGraphBuilder:
    def __init__(self, data_path="data", output_path="graph"):
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        
        # Data structures
        self.entity_dict = {}           # id -> name
        self.entity_name_to_id = {}     # name -> id  
        self.embeddings = None
        self.graph = nx.DiGraph()       # 有向图
        self.relation_types = set()
        
    def load_entity_dict(self):
        """Load entity dictionary from kb_entity_dict.txt"""
        print("Loading entity dictionary...")
        
        entity_file = self.data_path / "kb_entity_dict.txt"
        with open(entity_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '\t' in line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        idx = int(parts[0])
                        name = parts[1]
                        self.entity_dict[idx] = name
                        self.entity_name_to_id[name] = idx
        
        print(f"Loaded {len(self.entity_dict)} entities")
        return self
    
    def load_embeddings(self):
        """Load entity embeddings from kb_entity.npz"""
        print("Loading embeddings...")
        
        embedding_file = self.data_path / "kb_entity.npz"
        if embedding_file.exists():
            self.embeddings = np.load(embedding_file)
            print(f"Loaded embeddings for {len(self.embeddings.files)} entities")
        else:
            print("No embeddings file found")
        
        return self
    
    def load_relations(self):
        """Load relations from kb.txt"""
        print("Loading relations from kb.txt...")
        
        relation_file = self.data_path / "kb.txt"
        relations_loaded = 0
        
        with open(relation_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '|' in line:
                    parts = line.split('|')
                    if len(parts) == 3:
                        subject, relation, obj = parts
                        
                        # Get entity IDs
                        subject_id = self.entity_name_to_id.get(subject)
                        object_id = self.entity_name_to_id.get(obj)
                        
                        if subject_id is not None and object_id is not None:
                            self.graph.add_edge(subject_id, object_id, 
                                              relation=relation,
                                              subject=subject,
                                              object=obj)
                            self.relation_types.add(relation)
                            relations_loaded += 1
        
        print(f"Loaded {relations_loaded} relations")
        print(f"Relation types: {sorted(self.relation_types)}")
        return self
    
    def add_entity_nodes(self):
        """Add all entities as nodes to the graph"""
        print("Adding entity nodes...")
        
        nodes_added = 0
        for entity_id, entity_name in self.entity_dict.items():
            if not self.graph.has_node(entity_id):
                node_type = self._classify_entity(entity_name)
                self.graph.add_node(entity_id, 
                                  name=entity_name,
                                  type=node_type)
                nodes_added += 1
        
        print(f"Added {nodes_added} nodes to graph")
        print(f"Total graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        return self
    
    def _classify_entity(self, name):
        """Classify entity type based on name"""
        name_lower = name.lower()
        
        # Years
        if name.isdigit() and 1900 <= int(name) <= 2030:
            return 'year'
        
        # Common genres
        genres = {'drama', 'comedy', 'action', 'horror', 'thriller', 'romance', 
                 'war', 'documentary', 'adventure', 'crime', 'fantasy', 'mystery',
                 'sci-fi', 'western', 'animation', 'family', 'biography'}
        if name_lower in genres:
            return 'genre'
        
        # Languages
        languages = {'english', 'french', 'spanish', 'german', 'italian', 
                    'japanese', 'chinese', 'russian', 'portuguese', 'korean'}
        if name_lower in languages:
            return 'language'
        
        # Person names (heuristic: contains space and proper case)
        if (' ' in name and 
            len(name.split()) >= 2 and 
            any(word[0].isupper() for word in name.split())):
            return 'person'
        
        # Default to movie/entity
        return 'movie'
    
    def analyze_graph(self):
        """Analyze the constructed graph"""
        print("\n=== Graph Analysis ===")
        
        # Basic stats
        num_nodes = self.graph.number_of_nodes()
        num_edges = self.graph.number_of_edges()
        
        print(f"Nodes: {num_nodes:,}")
        print(f"Edges: {num_edges:,}")
        print(f"Relation types: {len(self.relation_types)}")
        
        # Node type distribution
        node_types = defaultdict(int)
        for _, data in self.graph.nodes(data=True):
            node_types[data.get('type', 'unknown')] += 1
        
        print(f"\nNode type distribution:")
        for node_type, count in sorted(node_types.items(), key=lambda x: -x[1]):
            print(f"  {node_type}: {count:,}")
        
        # Relation distribution
        relations = [data['relation'] for _, _, data in self.graph.edges(data=True)]
        relation_counts = Counter(relations)
        
        print(f"\nRelation distribution:")
        for relation, count in relation_counts.most_common():
            print(f"  {relation}: {count:,}")
        
        # Connectivity analysis
        print(f"\nConnectivity Analysis:")
        
        # Convert to undirected for connectivity analysis
        undirected = self.graph.to_undirected()
        
        # Connected components
        components = list(nx.connected_components(undirected))
        num_components = len(components)
        
        print(f"  Connected components: {num_components:,}")
        
        if components:
            # Largest component
            largest_component = max(components, key=len)
            largest_size = len(largest_component)
            
            print(f"  Largest component: {largest_size:,} nodes ({largest_size/num_nodes*100:.2f}%)")
            
            # Component size distribution
            comp_sizes = sorted([len(c) for c in components], reverse=True)
            print(f"  Component sizes (top 10): {comp_sizes[:10]}")
            
            # Isolated nodes
            isolated = [c for c in components if len(c) == 1]
            print(f"  Isolated nodes: {len(isolated):,}")
        
        # Degree analysis
        degrees = dict(self.graph.degree())
        in_degrees = dict(self.graph.in_degree()) 
        out_degrees = dict(self.graph.out_degree())
        
        if degrees:
            print(f"\nDegree Statistics:")
            print(f"  Average degree: {sum(degrees.values())/len(degrees):.2f}")
            print(f"  Average in-degree: {sum(in_degrees.values())/len(in_degrees):.2f}")
            print(f"  Average out-degree: {sum(out_degrees.values())/len(out_degrees):.2f}")
            print(f"  Max degree: {max(degrees.values()):,}")
            
            # Top nodes by degree
            top_degree = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]
            print(f"\n  Top 10 nodes by total degree:")
            for node_id, degree in top_degree:
                name = self.entity_dict.get(node_id, f'Entity-{node_id}')
                node_type = self.graph.nodes[node_id].get('type', 'unknown')
                print(f"    {name} ({node_type}): {degree}")
        
        return {
            'nodes': num_nodes,
            'edges': num_edges,
            'components': num_components,
            'largest_component': largest_size if components else 0,
            'isolated_nodes': len(isolated) if components else 0,
            'node_types': dict(node_types),
            'relation_types': list(self.relation_types)
        }
    
    def show_examples(self):
        """Show some example subgraphs"""
        print(f"\n=== Example Subgraphs ===")
        
        # Find some interesting nodes
        examples = []
        
        # Find a movie with many connections
        degrees = dict(self.graph.degree())
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
        
        for node_id, degree in top_nodes[:5]:
            node_type = self.graph.nodes[node_id].get('type', 'unknown')
            if node_type == 'movie' and degree > 10:
                examples.append(node_id)
                break
        
        # Find a person
        for node_id, data in self.graph.nodes(data=True):
            if data.get('type') == 'person':
                examples.append(node_id)
                break
        
        # Show examples
        for i, node_id in enumerate(examples[:2]):
            name = self.entity_dict.get(node_id, f'Entity-{node_id}')
            node_type = self.graph.nodes[node_id].get('type', 'unknown')
            
            print(f"\nExample {i+1}: {name} ({node_type})")
            
            # Show outgoing edges
            out_edges = list(self.graph.out_edges(node_id, data=True))[:5]
            if out_edges:
                print("  Outgoing relations:")
                for _, target, data in out_edges:
                    target_name = self.entity_dict.get(target, f'Entity-{target}')
                    relation = data['relation']
                    print(f"    {name} --[{relation}]--> {target_name}")
            
            # Show incoming edges  
            in_edges = list(self.graph.in_edges(node_id, data=True))[:5]
            if in_edges:
                print("  Incoming relations:")
                for source, _, data in in_edges:
                    source_name = self.entity_dict.get(source, f'Entity-{source}')
                    relation = data['relation']
                    print(f"    {source_name} --[{relation}]--> {name}")
    
    def save_graph(self):
        """Save the graph and related data"""
        print(f"\nSaving graph to {self.output_path}...")
        
        # 1. Save the main graph
        graph_file = self.output_path / "knowledge_graph.pkl"
        with open(graph_file, 'wb') as f:
            pickle.dump(self.graph, f)
        print(f"✓ Saved graph: {graph_file}")
        
        # 2. Save entity dictionary
        entity_file = self.output_path / "entity_dict.json"
        with open(entity_file, 'w', encoding='utf-8') as f:
            # Convert int keys to strings for JSON
            entity_dict_str = {str(k): v for k, v in self.entity_dict.items()}
            json.dump(entity_dict_str, f, ensure_ascii=False, indent=2)
        print(f"✓ Saved entities: {entity_file}")
        
        # 3. Save embeddings info
        if self.embeddings:
            embedding_info = {}
            for key in self.embeddings.files:
                try:
                    entity_id = int(key.split('-')[1])
                    if entity_id in self.entity_dict:
                        embedding = self.embeddings[key]
                        embedding_info[str(entity_id)] = {
                            'shape': list(embedding.shape),
                            'name': self.entity_dict[entity_id]
                        }
                except:
                    continue
            
            embedding_file = self.output_path / "embeddings_info.json"
            with open(embedding_file, 'w') as f:
                json.dump(embedding_info, f, indent=2)
            print(f"✓ Saved embedding info: {embedding_file}")
        
        # 4. Save graph statistics
        stats = self.analyze_graph()
        stats_file = self.output_path / "graph_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"✓ Saved statistics: {stats_file}")
        
        # 5. Save relation types
        relations_file = self.output_path / "relations.json"
        relations_data = {
            'relation_types': sorted(list(self.relation_types)),
            'total_relations': len(list(self.graph.edges()))
        }
        with open(relations_file, 'w') as f:
            json.dump(relations_data, f, indent=2)
        print(f"✓ Saved relations: {relations_file}")
        
        return self.output_path

def main():
    """Main function to build the complete knowledge graph"""
    print("=== Building Complete Knowledge Graph ===")
    print("Using files: kb_entity_dict.txt, kb_entity.npz, kb.txt")
    
    try:
        # Build the graph
        builder = CompleteKnowledgeGraphBuilder()
        
        builder.load_entity_dict()
        builder.load_embeddings() 
        builder.load_relations()
        builder.add_entity_nodes()
        
        # Analyze the graph
        stats = builder.analyze_graph()
        builder.show_examples()
        
        # Save the graph
        output_path = builder.save_graph()
        
        print(f"\n=== COMPLETE ===")
        print(f"✓ Knowledge graph built successfully!")
        print(f"✓ {stats['nodes']:,} nodes, {stats['edges']:,} edges")
        print(f"✓ {stats['components']:,} connected components")
        print(f"✓ Largest component: {stats['largest_component']:,} nodes ({stats['largest_component']/stats['nodes']*100:.1f}%)")
        print(f"✓ {stats['isolated_nodes']:,} isolated nodes")
        print(f"✓ {len(stats['relation_types'])} relation types")
        print(f"✓ Files saved to: {output_path}")
        
        return stats
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()