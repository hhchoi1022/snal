

from bridge.connector import GWPortalConnector
# %%
gwportal_connector = GWPortalConnector()
# %%
gwportal_connector.query_type = 'raw'



# %%
all_data = gwportal_connector.query(since_days = 1000)
# %%
all_tiles = set(all_data['object_name'])
# %%
filterset = []
for tile_id in all_tiles:
    tile_data = all_data[all_data['object_name'] == tile_id]
    filters = set(tile_data['filter'])
    
# %%
filterset
# %%
all_data = all_data.to_pandas()
import pandas as pd
#%%
# 전체 필터 목록

# all_filters = set(all_data['filter'].unique())
all_filters = set(['m400', 'm425', 'm450', 'm475', 'm500', 'm525', 'm550', 'm575', 'm600', 'm625', 'm650', 'm675', 'm700', 'm725', 'm750', 'm775', 'm800', 'm825', 'm850', 'm875'])

# 결과 저장용
results = []

# (날짜, 오브젝트) 기준 그룹화
grouped = all_data.groupby(['night', 'object_name'])

for (night, obj), group in grouped:
    
    observed_filters = set(group['filter'].unique())
    missing_filters = all_filters - observed_filters
    
    results.append({
        'night': night,
        'object_name': obj,
        'n_missing': len(missing_filters),
        'missing_filters': sorted(list(missing_filters))
    })

missing_df = pd.DataFrame(results)

missing_df = missing_df.sort_values(['night', 'object_name'])
missing_df
# %%
import pandas as pd
import numpy as np

all_data['date'] = (
    pd.to_datetime(
        all_data['obstime'],
        format='ISO8601',
        utc=True,
        errors='coerce'
    )
    .dt.tz_convert(None)   # timezone 제거
    .dt.date
)

filterset = all_filters
expected_filters = sorted(filterset)

rows = []

# grouped = all_data.groupby(['date'])
grouped = all_data.groupby('date')   # 리스트 제거

for date, group in grouped:
    observed = set(group['filter'].unique())
    for filt in expected_filters:
        rows.append({
            'date': date,
            'filter': filt,
            'missing': 0 if filt in observed else 1
        })

missing_matrix = pd.DataFrame(rows)
pivot = missing_matrix.pivot(index='filter', columns='date', values='missing')
# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

fig, ax = plt.subplots(figsize=(15, 6))

# white (0), red (1)
cmap = ListedColormap(['white', 'red'])
norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

im = ax.imshow(
    pivot.values,
    aspect='auto',
    cmap=cmap,
    norm=norm,
    interpolation='nearest'   # 🔥 이 줄 추가
)

# y축 필터 라벨
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index)

# 🔹 기본 얇은 grid
ax.set_yticks(np.arange(-0.5, len(pivot.index), 1), minor=True)
ax.grid(which='minor', axis='y', color='lightgray', linestyle=':', linewidth=2.0)
ax.tick_params(which='minor', bottom=False, left=False)

# 🔥 2칸마다 굵은 선 추가
for y in np.arange(-0.5, len(pivot.index), 2):
    ax.hlines(y, xmin=-0.5, xmax=len(pivot.columns)-0.5,
              colors='black', linewidth=1.0)

# xtick 간격 줄이기
step = 30
xticks = range(0, len(pivot.columns), step)
ax.set_xticks(xticks)
ax.set_xticklabels(
    [pivot.columns[i].strftime('%Y-%m-%d') for i in xticks],
    rotation=45,
    ha='right'
)

# colorbar
cbar = plt.colorbar(im, ticks=[0, 1])
cbar.ax.set_yticklabels(['Observed', 'Missing'])

plt.tight_layout()
plt.show()
# %%
exclude_filterlist = [
    ['m500'],
    ['m525'],
    ['m650'],
    ['m675'],
    ['m700'],
    ['m725'],
    ['m750'],
    ['m775'],
    ['m800'],
    ['m825'],
    ['m850'],
    ['m875'],
    ['m400', 'm550'],
    ['m425', 'm575'],
    ['m450', 'm600'],
    ['m475', 'm625'],
    ['m400', 'm425'],
    ['m450', 'm475'],
    ['m500', 'm525'],
    ['m550', 'm575'],
    ['m600', 'm625'],
    ['m650', 'm675'],
    ['m700', 'm725'],
    ['m750', 'm775'],
    ['m800', 'm825'],
    ['m850', 'm875'],
]
# %%
for exclude_filter in exclude_filterlist:
    filtered_filterset = sorted(list(filterset)).copy()
    for filter in exclude_filter:
        filtered_filterset.remove(filter)
    print(filtered_filterset)
# %%