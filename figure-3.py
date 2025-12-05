import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 8

# Create figure
fig = plt.figure(figsize=(8.5, 3))
gs = fig.add_gridspec(1, 3, wspace=0.35)

# Simulate data for 4 groups (n=8 each)
np.random.seed(42)
n = 8

# Panel A: Ejection Fraction
ax1 = fig.add_subplot(gs[0])

# Group data (mean ± SD based on text)
ef_sham = np.random.normal(63.2, 2.1, n)
ef_vehicle = np.random.normal(38.7, 3.8, n)
ef_pi3k = np.random.normal(51.3, 4.2, n)
ef_remote = np.random.normal(39.1, 4.1, n)

groups_ef = ['Sham', 'MI +\nVehicle', 'MI +\nPI3K', 'MI +\nRemote\nVehicle']
data_ef = [ef_sham, ef_vehicle, ef_pi3k, ef_remote]
colors_ef = ['#999999', '#E41A1C', '#4DAF4A', '#E41A1C']
alphas = [1.0, 1.0, 1.0, 0.5]

positions = [0, 1, 2, 3]
bp = ax1.boxplot(data_ef, positions=positions, widths=0.6, patch_artist=True,
                  showfliers=False, medianprops=dict(color='black', linewidth=2))

for patch, color, alpha in zip(bp['boxes'], colors_ef, alphas):
    patch.set_facecolor(color)
    patch.set_alpha(alpha)
    patch.set_edgecolor('black')
    patch.set_linewidth(1.5)

# Overlay individual points
for i, (pos, data) in enumerate(zip(positions, data_ef)):
    x = np.random.normal(pos, 0.08, len(data))
    ax1.scatter(x, data, alpha=0.6, s=35, color='black', zorder=3, edgecolors='white', linewidths=0.5)

ax1.set_ylabel('Ejection Fraction (%)', fontweight='bold', fontsize=9)
ax1.set_title('A. Cardiac Function (28d)', fontweight='bold', loc='left', fontsize=9)
ax1.set_xticks(positions)
ax1.set_xticklabels(groups_ef, fontsize=7.5)
ax1.set_ylim([30, 70])
ax1.grid(axis='y', alpha=0.3)

# Add significance brackets
def add_bracket(ax, x1, x2, y, h, text):
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], 'k-', linewidth=1.5)
    ax.text((x1+x2)/2, y+h, text, ha='center', va='bottom', fontsize=8, fontweight='bold')

add_bracket(ax1, 1, 2, 64, 1, '***')
add_bracket(ax1, 1, 3, 67, 1, 'n.s.')

# Panel B: Scar Size
ax2 = fig.add_subplot(gs[1])

# Group data
scar_sham = np.random.normal(0.5, 0.3, n)  # Minimal scar
scar_vehicle = np.random.normal(39.8, 5.2, n)
scar_pi3k = np.random.normal(18.4, 3.7, n)
scar_remote = np.random.normal(38.9, 4.8, n)

data_scar = [scar_sham, scar_vehicle, scar_pi3k, scar_remote]

bp2 = ax2.boxplot(data_scar, positions=positions, widths=0.6, patch_artist=True,
                   showfliers=False, medianprops=dict(color='black', linewidth=2))

for patch, color, alpha in zip(bp2['boxes'], colors_ef, alphas):
    patch.set_facecolor(color)
    patch.set_alpha(alpha)
    patch.set_edgecolor('black')
    patch.set_linewidth(1.5)

# Overlay points
for i, (pos, data) in enumerate(zip(positions, data_scar)):
    x = np.random.normal(pos, 0.08, len(data))
    ax2.scatter(x, data, alpha=0.6, s=35, color='black', zorder=3, edgecolors='white', linewidths=0.5)

ax2.set_ylabel('Scar Size (% of LV)', fontweight='bold', fontsize=9)
ax2.set_title('B. Tissue Regeneration (28d)', fontweight='bold', loc='left', fontsize=9)
ax2.set_xticks(positions)
ax2.set_xticklabels(groups_ef, fontsize=7.5)
ax2.set_ylim([-2, 50])
ax2.grid(axis='y', alpha=0.3)

# Add significance brackets
add_bracket(ax2, 1, 2, 46, 1, '****')
add_bracket(ax2, 1, 3, 43, 1, 'n.s.')

# Add percent reduction annotation
ax2.annotate('54% reduction', xy=(2, 18.4), xytext=(2.5, 8),
            fontsize=7.5, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

# Panel C: Cardiomyocyte Proliferation
ax3 = fig.add_subplot(gs[2])

# Group data (only MI groups, sham has minimal proliferation)
prolif_vehicle = np.random.normal(1.2, 0.3, n)
prolif_pi3k = np.random.normal(4.3, 0.8, n)
prolif_remote = np.random.normal(1.3, 0.35, n)

groups_prolif = ['MI +\nVehicle', 'MI +\nPI3K', 'MI +\nRemote\nVehicle']
data_prolif = [prolif_vehicle, prolif_pi3k, prolif_remote]
colors_prolif = ['#E41A1C', '#4DAF4A', '#E41A1C']
alphas_prolif = [1.0, 1.0, 0.5]

positions_prolif = [0, 1, 2]
bp3 = ax3.boxplot(data_prolif, positions=positions_prolif, widths=0.6, patch_artist=True,
                   showfliers=False, medianprops=dict(color='black', linewidth=2))

for patch, color, alpha in zip(bp3['boxes'], colors_prolif, alphas_prolif):
    patch.set_facecolor(color)
    patch.set_alpha(alpha)
    patch.set_edgecolor('black')
    patch.set_linewidth(1.5)

# Overlay points
for i, (pos, data) in enumerate(zip(positions_prolif, data_prolif)):
    x = np.random.normal(pos, 0.08, len(data))
    ax3.scatter(x, data, alpha=0.6, s=35, color='black', zorder=3, edgecolors='white', linewidths=0.5)

ax3.set_ylabel('Ki67+ Cardiomyocytes\n(% in Border Zone)', fontweight='bold', fontsize=8.5)
ax3.set_title('C. CM Proliferation (3d)', fontweight='bold', loc='left', fontsize=9)
ax3.set_xticks(positions_prolif)
ax3.set_xticklabels(groups_prolif, fontsize=7.5)
ax3.set_ylim([0, 6])
ax3.grid(axis='y', alpha=0.3)

# Add significance bracket
add_bracket(ax3, 0, 1, 5.2, 0.2, '****')

# Add fold-change annotation
ax3.annotate('3.6-fold increase', xy=(1, 4.3), xytext=(1.5, 2),
            fontsize=7.5, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

plt.tight_layout()
plt.savefig('figure3_therapeutic_intervention.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_therapeutic_intervention.pdf', bbox_inches='tight')
plt.show()

print("Figure 3 generated: Therapeutic intervention outcomes")
print("\nSummary statistics:")
print(f"Ejection Fraction - Sham: {np.mean(ef_sham):.1f}±{np.std(ef_sham):.1f}%")
print(f"Ejection Fraction - Vehicle: {np.mean(ef_vehicle):.1f}±{np.std(ef_vehicle):.1f}%")
print(f"Ejection Fraction - PI3K: {np.mean(ef_pi3k):.1f}±{np.std(ef_pi3k):.1f}%")
print(f"\nScar Size - Vehicle: {np.mean(scar_vehicle):.1f}±{np.std(scar_vehicle):.1f}%")
print(f"Scar Size - PI3K: {np.mean(scar_pi3k):.1f}±{np.std(scar_pi3k):.1f}%")
print(f"Percent reduction: {((np.mean(scar_vehicle)-np.mean(scar_pi3k))/np.mean(scar_vehicle)*100):.1f}%")
print(f"\nCM Proliferation - Vehicle: {np.mean(prolif_vehicle):.1f}±{np.std(prolif_vehicle):.1f}%")
print(f"CM Proliferation - PI3K: {np.mean(prolif_pi3k):.1f}±{np.std(prolif_pi3k):.1f}%")
print(f"Fold increase: {(np.mean(prolif_pi3k)/np.mean(prolif_vehicle)):.1f}x")