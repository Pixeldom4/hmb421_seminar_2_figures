import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch

# Set style
sns.set_style("white")
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 8

# Create figure with 3 panels
fig = plt.figure(figsize=(8.5, 3))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.2], wspace=0.35)

# Panel A: UMAP of all endocardial cells
ax1 = fig.add_subplot(gs[0])
np.random.seed(42)
n_cells = 5000

# Generate UMAP-like coordinates for different conditions
# P1 sham (cluster together)
p1_sham = np.random.multivariate_normal([2, 2], [[0.3, 0.1], [0.1, 0.3]], n_cells//6)
# P7 sham (cluster with P1 sham)
p7_sham = np.random.multivariate_normal([2.3, 1.8], [[0.3, 0.1], [0.1, 0.3]], n_cells//6)
# P1 MI 24h (regenerative trajectory)
p1_mi = np.random.multivariate_normal([5, 6], [[0.5, 0.2], [0.2, 0.5]], n_cells//3)
# P7 MI 24h (fibrotic trajectory)
p7_mi = np.random.multivariate_normal([8, 2], [[0.5, -0.2], [-0.2, 0.6]], n_cells//3)

ax1.scatter(p1_sham[:, 0], p1_sham[:, 1], c='#8DD3C7', s=2, alpha=0.6, label='P1 Sham', rasterized=True)
ax1.scatter(p7_sham[:, 0], p7_sham[:, 1], c='#BEBADA', s=2, alpha=0.6, label='P7 Sham', rasterized=True)
ax1.scatter(p1_mi[:, 0], p1_mi[:, 1], c='#FB8072', s=2, alpha=0.6, label='P1 MI 24h', rasterized=True)
ax1.scatter(p7_mi[:, 0], p7_mi[:, 1], c='#80B1D3', s=2, alpha=0.6, label='P7 MI 24h', rasterized=True)

ax1.set_xlabel('UMAP 1', fontweight='bold')
ax1.set_ylabel('UMAP 2', fontweight='bold')
ax1.set_title('A. Endocardial Cell Populations', fontweight='bold', loc='left', fontsize=9)
ax1.legend(frameon=False, fontsize=7, markerscale=3, loc='upper left')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_xticks([])
ax1.set_yticks([])

# Panel B: Trajectory inference showing bifurcation
ax2 = fig.add_subplot(gs[1])

# Create trajectory paths
t = np.linspace(0, 1, 100)
# Shared path (0-24h baseline)
shared_x = t * 2
shared_y = np.zeros_like(t) + 3

# Regenerative branch (P1, 24h onwards)
regen_t = np.linspace(0, 1, 100)
regen_x = 2 + regen_t * 3
regen_y = 3 + regen_t * 2.5

# Fibrotic branch (P7, 24h onwards)
fibro_t = np.linspace(0, 1, 100)
fibro_x = 2 + fibro_t * 3
fibro_y = 3 - fibro_t * 2.5

# Plot trajectories
ax2.plot(shared_x, shared_y, 'k-', linewidth=2.5, label='Baseline', zorder=1)
ax2.plot(regen_x, regen_y, color='#FB8072', linewidth=3, label='Regenerative (P1)', zorder=2)
ax2.plot(fibro_x, fibro_y, color='#80B1D3', linewidth=3, label='Fibrotic (P7)', zorder=2)

# Add arrow endpoints
ax2.arrow(regen_x[-2], regen_y[-2], regen_x[-1]-regen_x[-2], regen_y[-1]-regen_y[-2], 
          head_width=0.25, head_length=0.2, fc='#FB8072', ec='#FB8072', linewidth=0, zorder=3)
ax2.arrow(fibro_x[-2], fibro_y[-2], fibro_x[-1]-fibro_x[-2], fibro_y[-1]-fibro_y[-2], 
          head_width=0.25, head_length=0.2, fc='#80B1D3', ec='#80B1D3', linewidth=0, zorder=3)

# Mark branch point
ax2.scatter([2], [3], s=150, c='gold', edgecolors='black', linewidths=1.5, zorder=4, marker='*')
ax2.text(2, 2.3, 'Branch Point\n(24h post-MI)', ha='center', fontsize=7, fontweight='bold')

# Add pseudotime labels
ax2.text(0.2, 3.3, 'Sham', fontsize=7, ha='center')
ax2.text(4.5, 5.2, 'High MyD88\nPI3K/AKT active', fontsize=6.5, ha='center', 
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#FB8072', alpha=0.3))
ax2.text(4.5, 0.8, 'Low MyD88\nEndMT markers', fontsize=6.5, ha='center',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#80B1D3', alpha=0.3))

ax2.set_xlabel('Pseudotime', fontweight='bold')
ax2.set_ylabel('Regenerative ← → Fibrotic', fontweight='bold', fontsize=8)
ax2.set_title('B. Trajectory Bifurcation', fontweight='bold', loc='left', fontsize=9)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_xlim(-0.5, 5.5)
ax2.set_ylim(0, 6)

# Panel C: Pathway enrichment heatmap
ax3 = fig.add_subplot(gs[2])

pathways = ['PI3K/AKT\nSignaling', 'MyD88/TLR\nResponse', 'NRG1/ERBB\nSignaling', 
            'Proliferation', 'TGFβ\nSignaling', 'Collagen\nBiosynthesis', 'EndMT\nProgram']
p1_scores = np.array([2.41, 2.08, 2.18, 1.94, -0.82, -1.24, -2.15])
p7_scores = np.array([-1.87, -1.92, -1.45, -1.68, 2.63, 2.31, 2.44])

data = np.vstack([p1_scores, p7_scores]).T

im = ax3.imshow(data, cmap='RdBu_r', aspect='auto', vmin=-3, vmax=3)
ax3.set_xticks([0, 1])
ax3.set_xticklabels(['P1 MI\n(Regenerative)', 'P7 MI\n(Fibrotic)'], fontsize=7.5, fontweight='bold')
ax3.set_yticks(range(len(pathways)))
ax3.set_yticklabels(pathways, fontsize=7.5)
ax3.set_title('C. Pathway Enrichment (NES)', fontweight='bold', loc='left', fontsize=9)

# Add values as text
for i in range(len(pathways)):
    for j in range(2):
        text_color = 'white' if abs(data[i, j]) > 1.5 else 'black'
        ax3.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center', 
                color=text_color, fontsize=7, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
cbar.set_label('Normalized\nEnrichment Score', fontsize=7, fontweight='bold')
cbar.ax.tick_params(labelsize=6)

plt.tight_layout()
plt.savefig('figure1_endocardial_trajectories.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_endocardial_trajectories.pdf', bbox_inches='tight')
plt.show()

print("Figure 1 generated: Endocardial trajectories and pathway enrichment")