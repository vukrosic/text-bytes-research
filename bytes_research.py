#!/usr/bin/env python3
"""
Byte-Level Topological Data Analysis of Text Corpus
Analyzes text data using persistent homology on raw bytes with comprehensive visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from ripser import ripser
import warnings
from collections import Counter
from tqdm import tqdm
import pandas as pd
import os

warnings.filterwarnings('ignore')

# Configure plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class ByteTopologyAnalyzer:
    """Complete pipeline for byte-level topological analysis of text"""
    
    def __init__(self, num_documents=100, max_bytes=500):
        self.num_documents = num_documents
        self.max_bytes = max_bytes
        self.texts = []
        self.byte_arrays = []
        self.persistence_diagrams = []
        self.embeddings = {}
        self.output_folder = "byte_analysis_results"
        
    def load_data(self):
        """Load dataset from HuggingFace"""
        print("📚 Loading dataset from HuggingFace...")
        dataset = load_dataset(
            "HuggingFaceTB/smollm-corpus", 
            "cosmopedia-v2", 
            split="train", 
            streaming=True, 
            token=False
        )
        
        for i, item in enumerate(tqdm(dataset, total=self.num_documents, desc="Loading documents")):
            if i >= self.num_documents:
                break
            # Take first 3000 characters to ensure reasonable size
            self.texts.append(item["text"][:3000])
        
        print(f"✅ Loaded {len(self.texts)} documents")
        return self
    
    def text_to_bytes(self):
        """Convert texts to byte arrays"""
        print("\n🔤 Converting texts to byte arrays...")
        for text in tqdm(self.texts, desc="Converting to bytes"):
            # Convert to bytes and limit length
            byte_arr = np.array(list(text.encode('utf-8', errors='ignore')[:self.max_bytes]))
            self.byte_arrays.append(byte_arr)
        print(f"✅ Converted {len(self.byte_arrays)} texts to byte arrays")
        return self
    
    def create_embeddings(self):
        """Create sliding window embeddings from byte arrays"""
        print("\n🎯 Creating byte embeddings...")
        
        # Sliding window embedding (2D)
        print("  Creating sliding window embeddings...")
        window_embeddings = []
        for byte_arr in tqdm(self.byte_arrays, desc="Window embedding"):
            if len(byte_arr) < 2:
                window_embeddings.append(np.array([[0, 0]]))
                continue
            # Create 2D points from consecutive byte pairs
            points = np.array([[byte_arr[i], byte_arr[i+1]] 
                              for i in range(len(byte_arr)-1)])
            window_embeddings.append(points[:100])  # Limit points for computation
        self.embeddings['window'] = window_embeddings
        
        print("✅ Created embeddings")
        return self
    
    def compute_persistence(self):
        """Compute persistent homology for embeddings"""
        print("\n🔮 Computing persistent homology...")
        
        self.persistence_results = {}
        
        for embed_type, embeddings in self.embeddings.items():
            print(f"  Computing persistence for {embed_type} embeddings...")
            diagrams = []
            
            for points in tqdm(embeddings, desc=f"Computing {embed_type}"):
                try:
                    # Normalize points
                    if len(points) > 0:
                        scaler = StandardScaler()
                        points_normalized = scaler.fit_transform(points)
                        
                        # Compute persistence
                        result = ripser(points_normalized, maxdim=1)
                        diagrams.append(result['dgms'])
                    else:
                        diagrams.append([np.array([[0, 0]]), np.array([[0, 0]])])
                except:
                    # Fallback for any errors
                    diagrams.append([np.array([[0, 0]]), np.array([[0, 0]])])
            
            self.persistence_results[embed_type] = diagrams
        
        print("✅ Computed all persistence diagrams")
        return self

    def plot_byte_distribution(self, ax):
        all_bytes = np.concatenate(self.byte_arrays[:20])
        ax.hist(all_bytes, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax.set_title('Byte Value Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Byte Value (0-255)')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)

    def plot_byte_entropy(self, ax):
        entropies = []
        for byte_arr in self.byte_arrays:
            if len(byte_arr) > 0:
                _, counts = np.unique(byte_arr, return_counts=True)
                probs = counts / len(byte_arr)
                entropy = -np.sum(probs * np.log2(probs + 1e-10))
                entropies.append(entropy)
        
        ax.hist(entropies, bins=30, alpha=0.7, color='coral', edgecolor='black')
        ax.set_title('Byte Entropy Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Entropy (bits)')
        ax.set_ylabel('Number of Documents')
        ax.axvline(np.mean(entropies), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(entropies):.2f}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def plot_document_length_distribution(self, ax):
        lengths = [len(ba) for ba in self.byte_arrays]
        ax.hist(lengths, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        ax.set_title('Document Length Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Length (bytes)')
        ax.set_ylabel('Number of Documents')
        ax.axvline(np.mean(lengths), color='darkgreen', linestyle='--',
                    label=f'Mean: {np.mean(lengths):.0f}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def plot_byte_transition_heatmap(self, ax):
        transitions = np.zeros((256, 256))
        for byte_arr in self.byte_arrays[:10]:
            for i in range(len(byte_arr) - 1):
                transitions[byte_arr[i], byte_arr[i+1]] += 1
        
        # Show only ASCII printable range for clarity
        ascii_range = slice(32, 127)
        im = ax.imshow(np.log1p(transitions[ascii_range, ascii_range]), 
                       cmap='YlOrRd', aspect='auto')
        ax.set_title('Byte Transition Heatmap', fontsize=12, fontweight='bold')
        ax.set_xlabel('Next Byte')
        ax.set_ylabel('Current Byte')
        plt.colorbar(im, ax=ax, label='Log Frequency')

    def plot_persistence_diagram_sample(self, ax):
        dgm = self.persistence_results['window'][0]
        if len(dgm[0]) > 0:
            ax.scatter(dgm[0][:, 0], dgm[0][:, 1], alpha=0.6, s=30, c='blue', label='H0')
        ax.plot([0, np.max([1] + [d[1] for d in dgm[0] if len(d) > 0])], 
               [0, np.max([1] + [d[1] for d in dgm[0] if len(d) > 0])], 
               'k--', alpha=0.3)
        ax.set_title('Persistence Diagram (H0)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Birth')
        ax.set_ylabel('Death')
        ax.grid(True, alpha=0.3)
        ax.legend()

    def plot_topological_features_violin(self, ax):
        diagrams = self.persistence_results['window']
        num_h0_features = []
        max_persistence_h0 = []
        
        for dgm in diagrams:
            if len(dgm[0]) > 0:
                h0_pers = dgm[0][:, 1] - dgm[0][:, 0]
                num_h0_features.append(len(dgm[0]))
                max_persistence_h0.append(np.max(h0_pers) if len(h0_pers) > 0 else 0)
            else:
                num_h0_features.append(0)
                max_persistence_h0.append(0)
        
        data_to_plot = [num_h0_features, max_persistence_h0]
        labels = ['# H0 Features', 'Max Persistence']
        
        parts = ax.violinplot(data_to_plot, positions=range(len(labels)), 
                              showmeans=True, showmedians=True)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_title('Topological Features', fontsize=12, fontweight='bold')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        
        for pc in parts['bodies']:
            pc.set_facecolor('lightblue')
            pc.set_alpha(0.7)

    def plot_persistence_landscape(self, ax):
        diagrams = self.persistence_results['window']
        resolution = 100
        max_val = 2.0
        x = np.linspace(0, max_val, resolution)
        landscapes = []
        
        for dgm in diagrams[:20]:
            landscape = np.zeros(resolution)
            for point in dgm[0]:
                birth, death = point
                if death > birth:
                    for i, xi in enumerate(x):
                        if birth <= xi <= death:
                            height = min(xi - birth, death - xi)
                            landscape[i] = max(landscape[i], height)
            landscapes.append(landscape)
        
        if landscapes:
            mean_landscape = np.mean(landscapes, axis=0)
            std_landscape = np.std(landscapes, axis=0)
            
            ax.plot(x, mean_landscape, 'b-', linewidth=2, label='Mean')
            ax.fill_between(x, mean_landscape - std_landscape, 
                           mean_landscape + std_landscape, 
                           alpha=0.3, color='blue', label='±1 STD')
        
        ax.set_title('Persistence Landscape', fontsize=12, fontweight='bold')
        ax.set_xlabel('Filtration Value')
        ax.set_ylabel('Landscape Function')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def plot_document_similarity_heatmap(self, ax):
        diagrams = self.persistence_results['window']
        feature_vectors = []
        
        for dgm in diagrams[:30]:
            features = []
            if len(dgm[0]) > 0:
                h0_pers = dgm[0][:, 1] - dgm[0][:, 0]
                features.extend([
                    len(dgm[0]),
                    np.nan_to_num(np.mean(h0_pers)),
                    np.nan_to_num(np.max(h0_pers)),
                    np.nan_to_num(np.std(h0_pers))
                ])
            else:
                features.extend([0, 0, 0, 0])
            
            if len(dgm) > 1 and len(dgm[1]) > 0:
                h1_pers = dgm[1][:, 1] - dgm[1][:, 0]
                features.extend([
                    len(dgm[1]),
                    np.nan_to_num(np.mean(h1_pers)) if len(h1_pers) > 0 else 0,
                    np.nan_to_num(np.max(h1_pers)) if len(h1_pers) > 0 else 0,
                    np.nan_to_num(np.std(h1_pers)) if len(h1_pers) > 0 else 0
                ])
            else:
                features.extend([0, 0, 0, 0])
            
            feature_vectors.append(features)
        
        if feature_vectors:
            feature_vectors = np.array(feature_vectors)
            scaler = StandardScaler()
            feature_vectors_norm = scaler.fit_transform(feature_vectors)
            dist_matrix = squareform(pdist(feature_vectors_norm, metric='euclidean'))
            
            im = ax.imshow(dist_matrix, cmap='RdYlBu_r', aspect='auto')
            ax.set_title('Document Similarity Matrix', fontsize=12, fontweight='bold')
            ax.set_xlabel('Document Index')
            ax.set_ylabel('Document Index')
            plt.colorbar(im, ax=ax, label='Distance')

    def plot_feature_correlation_heatmap(self, ax):
        feature_matrix = []
        feature_names = ['Byte Entropy', 'Doc Length', 'Mean H0', 'Mean H1', 'Max Persistence']
        
        for i in range(min(50, len(self.byte_arrays))):
            row = []
            
            # Byte entropy
            if len(self.byte_arrays[i]) > 0:
                _, counts = np.unique(self.byte_arrays[i], return_counts=True)
                probs = counts / len(self.byte_arrays[i])
                entropy = -np.sum(probs * np.log2(probs + 1e-10))
                row.append(entropy)
            else:
                row.append(0)
            
            # Document length
            row.append(len(self.byte_arrays[i]))
            
            # Topological features
            dgm = self.persistence_results['window'][i]
            
            if len(dgm[0]) > 0:
                row.append(np.mean(dgm[0][:, 1] - dgm[0][:, 0]))
            else:
                row.append(0)
            
            if len(dgm) > 1 and len(dgm[1]) > 0:
                row.append(np.mean(dgm[1][:, 1] - dgm[1][:, 0]))
            else:
                row.append(0)
            
            max_pers = 0
            if len(dgm[0]) > 0:
                max_pers = max(max_pers, np.max(dgm[0][:, 1] - dgm[0][:, 0]))
            if len(dgm) > 1 and len(dgm[1]) > 0:
                max_pers = max(max_pers, np.max(dgm[1][:, 1] - dgm[1][:, 0]))
            row.append(max_pers)
            
            feature_matrix.append(row)
        
        if feature_matrix:
            feature_matrix = np.array(feature_matrix)
            corr_matrix = np.corrcoef(feature_matrix.T)
            
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', 
                       xticklabels=feature_names, yticklabels=feature_names,
                       cmap='coolwarm', center=0, ax=ax)
            ax.set_title('Feature Correlation Matrix', fontsize=12, fontweight='bold')

    def plot_byte_frequency_spectrum(self, ax):
        all_byte_freqs = np.zeros(256)
        for byte_arr in self.byte_arrays:
            unique, counts = np.unique(byte_arr, return_counts=True)
            for val, count in zip(unique, counts):
                all_byte_freqs[val] += count
        
        all_byte_freqs = all_byte_freqs / np.sum(all_byte_freqs)
        
        ax.bar(range(256), all_byte_freqs, color='steelblue', alpha=0.7)
        ax.set_title('Global Byte Frequency Spectrum', fontsize=12, fontweight='bold')
        ax.set_xlabel('Byte Value')
        ax.set_ylabel('Relative Frequency')
        ax.set_xlim([0, 255])
        
        # Highlight ASCII printable range
        ax.axvspan(32, 126, alpha=0.2, color='green', label='ASCII Printable')
        ax.legend()

    def plot_summary_statistics(self, ax):
        ax.axis('off')
        
        # Calculate summary statistics
        total_bytes = sum([len(ba) for ba in self.byte_arrays])
        avg_length = np.mean([len(ba) for ba in self.byte_arrays])
        entropies = []
        for byte_arr in self.byte_arrays:
            if len(byte_arr) > 0:
                _, counts = np.unique(byte_arr, return_counts=True)
                probs = counts / len(byte_arr)
                entropy = -np.sum(probs * np.log2(probs + 1e-10))
                entropies.append(entropy)
        avg_entropy = np.mean(entropies)
        
        h0_counts = [len(dgm[0]) for dgm in self.persistence_results['window']]
        h1_counts = [len(dgm[1]) if len(dgm) > 1 else 0 for dgm in self.persistence_results['window']]
        avg_h0 = np.mean(h0_counts)
        avg_h1 = np.mean(h1_counts)
        
        summary_text = f"""
        📊 ANALYSIS SUMMARY
        
        Documents analyzed: {len(self.texts)}
        Total bytes processed: {total_bytes:,}
        Average document length: {avg_length:.0f} bytes
        Average byte entropy: {avg_entropy:.2f} bits
        
        🔮 TOPOLOGICAL FEATURES
        Average H0 features (connected components): {avg_h0:.1f} ± {np.std(h0_counts):.1f}
        Average H1 features (loops): {avg_h1:.1f} ± {np.std(h1_counts):.1f}
        
        💾 Results saved to: {self.output_folder}/
        """
        
        ax.text(0.5, 0.5, summary_text, transform=ax.transAxes, 
                 fontsize=14, ha='center', va='center', 
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    def create_comprehensive_plot(self):
        """Create one comprehensive plot combining key insights"""
        print("\n🎨 Creating comprehensive visualization...")
        
        # Create output folder
        output_folder = "byte_analysis_results"
        os.makedirs(output_folder, exist_ok=True)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle('🔍 Byte-Level Topological Data Analysis - Comprehensive Overview', 
                     fontsize=20, fontweight='bold', y=0.98)
        
        # Create grid layout
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Byte distribution (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        all_bytes = np.concatenate(self.byte_arrays[:20])
        ax1.hist(all_bytes, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.set_title('Byte Value Distribution', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Byte Value (0-255)')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # 2. Byte entropy per document (top middle-left)
        ax2 = fig.add_subplot(gs[0, 1])
        entropies = []
        for byte_arr in self.byte_arrays:
            if len(byte_arr) > 0:
                _, counts = np.unique(byte_arr, return_counts=True)
                probs = counts / len(byte_arr)
                entropy = -np.sum(probs * np.log2(probs + 1e-10))
                entropies.append(entropy)
        
        ax2.hist(entropies, bins=30, alpha=0.7, color='coral', edgecolor='black')
        ax2.set_title('Byte Entropy Distribution', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Entropy (bits)')
        ax2.set_ylabel('Number of Documents')
        ax2.axvline(np.mean(entropies), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(entropies):.2f}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Document length distribution (top middle-right)
        ax3 = fig.add_subplot(gs[0, 2])
        lengths = [len(ba) for ba in self.byte_arrays]
        ax3.hist(lengths, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        ax3.set_title('Document Length Distribution', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Length (bytes)')
        ax3.set_ylabel('Number of Documents')
        ax3.axvline(np.mean(lengths), color='darkgreen', linestyle='--',
                    label=f'Mean: {np.mean(lengths):.0f}')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Byte transition heatmap (top right)
        ax4 = fig.add_subplot(gs[0, 3])
        transitions = np.zeros((256, 256))
        for byte_arr in self.byte_arrays[:10]:
            for i in range(len(byte_arr) - 1):
                transitions[byte_arr[i], byte_arr[i+1]] += 1
        
        # Show only ASCII printable range for clarity
        ascii_range = slice(32, 127)
        im = ax4.imshow(np.log1p(transitions[ascii_range, ascii_range]), 
                       cmap='YlOrRd', aspect='auto')
        ax4.set_title('Byte Transition Heatmap', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Next Byte')
        ax4.set_ylabel('Current Byte')
        plt.colorbar(im, ax=ax4, label='Log Frequency')
        
        # 5. Persistence diagram sample (middle left)
        ax5 = fig.add_subplot(gs[1, 0])
        dgm = self.persistence_results['window'][0]
        if len(dgm[0]) > 0:
            ax5.scatter(dgm[0][:, 0], dgm[0][:, 1], alpha=0.6, s=30, c='blue', label='H0')
        ax5.plot([0, np.max([1] + [d[1] for d in dgm[0] if len(d) > 0])], 
               [0, np.max([1] + [d[1] for d in dgm[0] if len(d) > 0])], 
               'k--', alpha=0.3)
        ax5.set_title('Persistence Diagram (H0)', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Birth')
        ax5.set_ylabel('Death')
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        
        # 6. Topological features violin plot (middle middle-left)
        ax6 = fig.add_subplot(gs[1, 1])
        diagrams = self.persistence_results['window']
        num_h0_features = []
        max_persistence_h0 = []
        
        for dgm in diagrams:
            if len(dgm[0]) > 0:
                h0_pers = dgm[0][:, 1] - dgm[0][:, 0]
                num_h0_features.append(len(dgm[0]))
                max_persistence_h0.append(np.max(h0_pers) if len(h0_pers) > 0 else 0)
            else:
                num_h0_features.append(0)
                max_persistence_h0.append(0)
        
        data_to_plot = [num_h0_features, max_persistence_h0]
        labels = ['# H0 Features', 'Max Persistence']
        
        parts = ax6.violinplot(data_to_plot, positions=range(len(labels)), 
                              showmeans=True, showmedians=True)
        ax6.set_xticks(range(len(labels)))
        ax6.set_xticklabels(labels)
        ax6.set_title('Topological Features', fontsize=12, fontweight='bold')
        ax6.set_ylabel('Value')
        ax6.grid(True, alpha=0.3)
        
        for pc in parts['bodies']:
            pc.set_facecolor('lightblue')
            pc.set_alpha(0.7)
        
        # 7. Persistence landscape (middle middle-right)
        ax7 = fig.add_subplot(gs[1, 2])
        resolution = 100
        max_val = 2.0
        x = np.linspace(0, max_val, resolution)
        landscapes = []
        
        for dgm in diagrams[:20]:
            landscape = np.zeros(resolution)
            for point in dgm[0]:
                birth, death = point
                if death > birth:
                    for i, xi in enumerate(x):
                        if birth <= xi <= death:
                            height = min(xi - birth, death - xi)
                            landscape[i] = max(landscape[i], height)
            landscapes.append(landscape)
        
        if landscapes:
            mean_landscape = np.mean(landscapes, axis=0)
            std_landscape = np.std(landscapes, axis=0)
            
            ax7.plot(x, mean_landscape, 'b-', linewidth=2, label='Mean')
            ax7.fill_between(x, mean_landscape - std_landscape, 
                           mean_landscape + std_landscape, 
                           alpha=0.3, color='blue', label='±1 STD')
        
        ax7.set_title('Persistence Landscape', fontsize=12, fontweight='bold')
        ax7.set_xlabel('Filtration Value')
        ax7.set_ylabel('Landscape Function')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8. Document similarity heatmap (middle right)
        ax8 = fig.add_subplot(gs[1, 3])
        feature_vectors = []
        
        for dgm in diagrams[:30]:
            features = []
            if len(dgm[0]) > 0:
                h0_pers = dgm[0][:, 1] - dgm[0][:, 0]
                features.extend([
                    len(dgm[0]),
                    np.nan_to_num(np.mean(h0_pers)),
                    np.nan_to_num(np.max(h0_pers)),
                    np.nan_to_num(np.std(h0_pers))
                ])
            else:
                features.extend([0, 0, 0, 0])
            
            if len(dgm) > 1 and len(dgm[1]) > 0:
                h1_pers = dgm[1][:, 1] - dgm[1][:, 0]
                features.extend([
                    len(dgm[1]),
                    np.nan_to_num(np.mean(h1_pers)) if len(h1_pers) > 0 else 0,
                    np.nan_to_num(np.max(h1_pers)) if len(h1_pers) > 0 else 0,
                    np.nan_to_num(np.std(h1_pers)) if len(h1_pers) > 0 else 0
                ])
            else:
                features.extend([0, 0, 0, 0])
            
            feature_vectors.append(features)
        
        if feature_vectors:
            feature_vectors = np.array(feature_vectors)
            scaler = StandardScaler()
            feature_vectors_norm = scaler.fit_transform(feature_vectors)
            dist_matrix = squareform(pdist(feature_vectors_norm, metric='euclidean'))
            
            im = ax8.imshow(dist_matrix, cmap='RdYlBu_r', aspect='auto')
            ax8.set_title('Document Similarity Matrix', fontsize=12, fontweight='bold')
            ax8.set_xlabel('Document Index')
            ax8.set_ylabel('Document Index')
            plt.colorbar(im, ax=ax8, label='Distance')
        
        # 9. Feature correlation heatmap (bottom left, spanning 2 columns)
        ax9 = fig.add_subplot(gs[2, :2])
        
        feature_matrix = []
        feature_names = ['Byte Entropy', 'Doc Length', 'Mean H0', 'Mean H1', 'Max Persistence']
        
        for i in range(min(50, len(self.byte_arrays))):
            row = []
            
            # Byte entropy
            if len(self.byte_arrays[i]) > 0:
                _, counts = np.unique(self.byte_arrays[i], return_counts=True)
                probs = counts / len(self.byte_arrays[i])
                entropy = -np.sum(probs * np.log2(probs + 1e-10))
                row.append(entropy)
            else:
                row.append(0)
            
            # Document length
            row.append(len(self.byte_arrays[i]))
            
            # Topological features
            dgm = self.persistence_results['window'][i]
            
            if len(dgm[0]) > 0:
                row.append(np.mean(dgm[0][:, 1] - dgm[0][:, 0]))
            else:
                row.append(0)
            
            if len(dgm) > 1 and len(dgm[1]) > 0:
                row.append(np.mean(dgm[1][:, 1] - dgm[1][:, 0]))
            else:
                row.append(0)
            
            max_pers = 0
            if len(dgm[0]) > 0:
                max_pers = max(max_pers, np.max(dgm[0][:, 1] - dgm[0][:, 0]))
            if len(dgm) > 1 and len(dgm[1]) > 0:
                max_pers = max(max_pers, np.max(dgm[1][:, 1] - dgm[1][:, 0]))
            row.append(max_pers)
            
            feature_matrix.append(row)
        
        if feature_matrix:
            feature_matrix = np.array(feature_matrix)
            corr_matrix = np.corrcoef(feature_matrix.T)
            
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', 
                       xticklabels=feature_names, yticklabels=feature_names,
                       cmap='coolwarm', center=0, ax=ax9)
            ax9.set_title('Feature Correlation Matrix', fontsize=12, fontweight='bold')
        
        # 10. Byte frequency spectrum (bottom middle-right, spanning 2 columns)
        ax10 = fig.add_subplot(gs[2, 2:])
        
        all_byte_freqs = np.zeros(256)
        for byte_arr in self.byte_arrays:
            unique, counts = np.unique(byte_arr, return_counts=True)
            for val, count in zip(unique, counts):
                all_byte_freqs[val] += count
        
        all_byte_freqs = all_byte_freqs / np.sum(all_byte_freqs)
        
        ax10.bar(range(256), all_byte_freqs, color='steelblue', alpha=0.7)
        ax10.set_title('Global Byte Frequency Spectrum', fontsize=12, fontweight='bold')
        ax10.set_xlabel('Byte Value')
        ax10.set_ylabel('Relative Frequency')
        ax10.set_xlim([0, 255])
        
        # Highlight ASCII printable range
        ax10.axvspan(32, 126, alpha=0.2, color='green', label='ASCII Printable')
        ax10.legend()
        
        # 11. Summary statistics (bottom, spanning full width)
        ax11 = fig.add_subplot(gs[3, :])
        ax11.axis('off')
        
        # Calculate summary statistics
        total_bytes = sum([len(ba) for ba in self.byte_arrays])
        avg_length = np.mean([len(ba) for ba in self.byte_arrays])
        avg_entropy = np.mean(entropies)
        
        h0_counts = [len(dgm[0]) for dgm in self.persistence_results['window']]
        h1_counts = [len(dgm[1]) if len(dgm) > 1 else 0 for dgm in self.persistence_results['window']]
        avg_h0 = np.mean(h0_counts)
        avg_h1 = np.mean(h1_counts)
        
        summary_text = f"""
        📊 ANALYSIS SUMMARY
        
        Documents analyzed: {len(self.texts)}
        Total bytes processed: {total_bytes:,}
        Average document length: {avg_length:.0f} bytes
        Average byte entropy: {avg_entropy:.2f} bits
        
        🔮 TOPOLOGICAL FEATURES
        Average H0 features (connected components): {avg_h0:.1f} ± {np.std(h0_counts):.1f}
        Average H1 features (loops): {avg_h1:.1f} ± {np.std(h1_counts):.1f}
        
        💾 Results saved to: {output_folder}/
        """
        
        ax11.text(0.5, 0.5, summary_text, transform=ax11.transAxes, 
                 fontsize=14, ha='center', va='center', 
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        # Save the plot
        output_path = os.path.join(output_folder, "byte_topology_analysis.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Comprehensive visualization saved to: {output_path}")
        
        plt.close()  # Close the figure to free memory
        
        return output_path

    def save_individual_plots(self):
        """Save individual plots to files"""
        print("\n💾 Saving individual plots...")
        os.makedirs(self.output_folder, exist_ok=True)

        plot_functions = {
            "byte_distribution": self.plot_byte_distribution,
            "byte_entropy": self.plot_byte_entropy,
            "document_length_distribution": self.plot_document_length_distribution,
            "byte_transition_heatmap": self.plot_byte_transition_heatmap,
            "persistence_diagram_sample": self.plot_persistence_diagram_sample,
            "topological_features_violin": self.plot_topological_features_violin,
            "persistence_landscape": self.plot_persistence_landscape,
            "document_similarity_heatmap": self.plot_document_similarity_heatmap,
            "feature_correlation_heatmap": self.plot_feature_correlation_heatmap,
            "byte_frequency_spectrum": self.plot_byte_frequency_spectrum,
        }

        for name, plot_func in plot_functions.items():
            fig, ax = plt.subplots(figsize=(8, 6))
            plot_func(ax)
            output_path = os.path.join(self.output_folder, f"{name}.png")
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  ✅ Saved {name}.png")
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("\n" + "="*60)
        print("🚀 BYTE-LEVEL TOPOLOGICAL DATA ANALYSIS")
        print("="*60 + "\n")
        
        # Load and prepare data
        self.load_data()
        self.text_to_bytes()
        self.create_embeddings()
        self.compute_persistence()
        
        print("\n📊 Generating comprehensive visualization...\n")
        
        # Create and save the comprehensive plot
        output_path = self.create_comprehensive_plot()
        self.save_individual_plots()
        
        print("\n" + "="*60)
        print("✅ ANALYSIS COMPLETE!")
        print(f"📁 Results saved to: {output_path}")
        print("="*60 + "\n")
        
        # Print summary statistics
        self.print_summary()
    
    def print_summary(self):
        """Print summary statistics"""
        print("📈 SUMMARY STATISTICS")
        print("-" * 40)
        print(f"Documents analyzed: {len(self.texts)}")
        print(f"Average document length: {np.mean([len(ba) for ba in self.byte_arrays]):.0f} bytes")
        print(f"Total bytes processed: {sum([len(ba) for ba in self.byte_arrays]):,}")
        
        print("\n🔮 TOPOLOGICAL FEATURES")
        print("-" * 40)
        
        for embed_type, diagrams in self.persistence_results.items():
            h0_counts = [len(dgm[0]) for dgm in diagrams]
            h1_counts = [len(dgm[1]) if len(dgm) > 1 else 0 for dgm in diagrams]
            
            print(f"\n{embed_type.upper()} Embedding:")
            print(f"  Avg H0 features: {np.mean(h0_counts):.1f} ± {np.std(h0_counts):.1f}")
            print(f"  Avg H1 features: {np.mean(h1_counts):.1f} ± {np.std(h1_counts):.1f}")

# Main execution
if __name__ == "__main__":
    # Configure analysis parameters
    analyzer = ByteTopologyAnalyzer(
        num_documents=100,  # Adjust based on your computational resources
        max_bytes=500       # Limit bytes per document for faster computation
    )
    
    # Run complete analysis
    analyzer.run_complete_analysis()