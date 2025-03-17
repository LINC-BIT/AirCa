from flask import Flask, render_template, jsonify, request
import folium
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

# 模拟城市数据
cities = [
    {'name': 'SHA', 'lat': 31.1746, 'lon': 121.3992, 'avg': 5.08},
    {'name': 'PEK', 'lat': 39.9075, 'lon': 116.3972, 'avg': 4.21},
    {'name': 'CAN', 'lat': 23.1291, 'lon': 113.2644, 'avg': 3.76},
    {'name': 'SYX', 'lat': 18.2432, 'lon': 109.5128, 'avg': 2.93},
    {'name': 'MUC', 'lat': 48.1374, 'lon': 11.5766, 'avg': 6.15},
    {'name': 'SZX', 'lat': 22.5431, 'lon': 114.0579, 'avg': 4.98},
    {'name': 'FRA', 'lat': 50.0334, 'lon': 8.5704, 'avg': 5.32},
    {'name': 'CDG', 'lat': 49.0127, 'lon': 2.5507, 'avg': 3.89},
    {'name': 'HKG', 'lat': 22.3087, 'lon': 114.1740, 'avg': 7.01},
    {'name': 'GRU', 'lat': -23.4355, 'lon': -46.4854, 'avg': 2.67},
    {'name': 'CKG', 'lat': 29.5593, 'lon': 106.5528, 'avg': 3.14},
    {'name': 'CTU', 'lat': 30.5753, 'lon': 104.0657, 'avg': 4.56},
    {'name': 'URC', 'lat': 43.7928, 'lon': 87.6177, 'avg': 1.99},
    {'name': 'SIN', 'lat': 1.3521, 'lon': 103.8198, 'avg': 5.88},
    {'name': 'HAK', 'lat': 20.0319, 'lon': 110.3524, 'avg': 2.22},
    {'name': 'LHR', 'lat': 51.4778, 'lon': -0.4613, 'avg': 6.33},
    {'name': 'WUH', 'lat': 30.5848, 'lon': 114.2933, 'avg': 3.55},
    {'name': 'IAD', 'lat': 38.9551, 'lon': -77.4564, 'avg': 2.88},
    {'name': 'SVO', 'lat': 55.9525, 'lon': 37.4124, 'avg': 4.77},
    {'name': 'FCO', 'lat': 41.8043, 'lon': 12.2954, 'avg': 3.33},
    {'name': 'LAX', 'lat': 33.9425, 'lon': -118.4080, 'avg': 7.11},
    {'name': 'SFO', 'lat': 37.6189, 'lon': -122.3750, 'avg': 5.66},
    {'name': 'YVR', 'lat': 49.1937, 'lon': -123.1933, 'avg': 2.44},
    {'name': 'BKK', 'lat': 13.7367, 'lon': 100.5231, 'avg': 3.66},
    {'name': 'MAD', 'lat': 40.4516, 'lon': -3.5655, 'avg': 4.11},
    {'name': 'HGH', 'lat': 30.2587, 'lon': 120.1535, 'avg': 2.77},
    {'name': 'DAT', 'lat': 40.0966, 'lon': 113.3081, 'avg': 1.66},
    {'name': 'CHG', 'lat': 41.5475, 'lon': 120.4464, 'avg': 3.99},
    {'name': 'DPS', 'lat': -8.6503, 'lon': 115.2124, 'avg': 2.11},
    {'name': 'SYD', 'lat': -33.9425, 'lon': 151.1778, 'avg': 5.44}
]

# 城市运输数据
city_data = {
    'SHA': {'weight': [1800, 1900, 2000, 2100, 2200], 'volume': [90, 95, 100, 105, 110], 'length': [260, 270, 280, 290, 300], 'width': [180, 185, 190, 195, 200], 'height': [130, 135, 140, 145, 150]},
    'PEK': {'weight': [1600, 1700, 1800, 1900, 2000], 'volume': [80, 82, 84, 86, 88], 'length': [240, 250, 260, 270, 280], 'width': [170, 172, 174, 176, 178], 'height': [120, 122, 124, 126, 128]},
    'CAN': {'weight': [1400, 1500, 1600, 1700, 1800], 'volume': [70, 72, 74, 76, 78], 'length': [220, 230, 240, 250, 260], 'width': [160, 162, 164, 166, 168], 'height': [110, 112, 114, 116, 118]},
    'SYX': {'weight': [1200, 1300, 1400, 1500, 1600], 'volume': [60, 62, 64, 66, 68], 'length': [200, 210, 220, 230, 240], 'width': [150, 152, 154, 156, 158], 'height': [100, 102, 104, 106, 108]},
    'MUC': {'weight': [2000, 2100, 2200, 2300, 2400], 'volume': [100, 102, 104, 106, 108], 'length': [280, 290, 300, 310, 320], 'width': [190, 192, 194, 196, 198], 'height': [140, 142, 144, 146, 148]},
    'SZX': {'weight': [1700, 1800, 1900, 2000, 2100], 'volume': [75, 77, 79, 81, 83], 'length': [230, 240, 250, 260, 270], 'width': [165, 167, 169, 171, 173], 'height': [115, 117, 119, 121, 123]},
    'FRA': {'weight': [2200, 2300, 2400, 2500, 2600], 'volume': [110, 112, 114, 116, 118], 'length': [300, 310, 320, 330, 340], ' width': [200, 202, 204, 206, 208], 'height': [150, 152, 154, 156, 158]},
    'CDG': {'weight': [2100, 2200, 2300, 2400, 2500], 'volume': [105, 107, 109, 111, 113], 'length': [290, 300, 310, 320, 330], 'width': [195, 197, 199, 201, 203], 'height': [145, 147, 149, 151, 153]},
    'HKG': {'weight': [1900, 2000, 2100, 2200, 2300], 'volume': [95, 97, 99, 101, 103], 'length': [270, 280, 290, 300, 310], 'width': [185, 187, 189, 191, 193], 'height': [135, 137, 139, 141, 143]},
    'GRU': {'weight': [2300, 2400, 2500, 2600, 2700], 'volume': [115, 117, 119, 121, 123], 'length': [310, 320, 330, 340, 350], 'width': [205, 207, 209, 211, 213], 'height': [155, 157, 159, 161, 163]},
    'CKG': {'weight': [1500, 1600, 1700, 1800, 1900], 'volume': [72, 74, 76, 78, 80], 'length': [225, 235, 245, 255, 265], 'width': [162, 164, 166, 168, 170], 'height': [112, 114, 116, 118, 120]},
    'CTU': {'weight': [1650, 1750, 1850, 1950, 2050], 'volume': [78, 80, 82, 84, 86], 'length': [245, 255, 265, 275, 285], 'width': [172, 174, 176, 178, 180], 'height': [122, 124, 126, 128, 130]},
    'URC': {'weight': [1450, 1550, 1650, 1750, 1850], 'volume': [68, 70, 72, 74, 76], 'length': [215, 225, 235, 245, 255], 'width': [158, 160, 162, 164, 166], 'height': [108, 110, 112, 114, 116]},
    'SIN': {'weight': [1300, 1400, 1500, 1600, 1700], 'volume': [65, 67, 69, 71, 73], 'length': [205, 215, 225, 235, 245], 'width': [155, 157, 159, 161, 163], 'height': [105, 107, 109, 111, 113]},
    'HAK': {'weight': [1150, 1250, 1350, 1450, 1550], 'volume': [62, 64, 66, 68, 70], 'length': [200, 210, 220, 230, 240], 'width': [152, 154, 156, 158, 160], 'height': [102, 104, 106, 108, 110]},
    'LHR': {'weight': [2400, 2500, 2600, 2700, 2800], 'volume': [120, 122, 124, 126, 128], 'length': [320, 330, 340, 350, 360], 'width': [210, 212, 214, 216, 218], 'height': [160, 162, 164, 166, 168]},
    'WUH': {'weight': [1750, 1850, 1950, 2050, 2150], 'volume': [82, 84, 86, 88, 90], 'length': [255, 265, 275, 285, 295], 'width': [178, 180, 182, 184, 186], 'height': [128, 130, 132, 134, 136]},
    'IAD': {'weight': [2500, 2600, 2700, 2800, 2900], 'volume': [125, 127, 129, 131, 133], 'length': [330, 340, 350, 360, 370], 'width': [215, 217, 219, 221, 223], 'height': [165, 167, 169, 171, 173]},
    'SVO': {'weight': [2050, 2150, 2250, 2350, 2450], 'volume': [108, 110, 112, 114, 116], 'length': [295, 305, 315, 325, 335], 'width': [198, 200, 202, 204, 206], 'height': [148, 150, 152, 154, 156]},
    'FCO': {'weight': [2150, 2250, 2350, 2450, 2550], 'volume': [112, 114, 116, 118, 120], 'length': [305, 315, 325, 335, 345], 'width': [202, 204, 206, 208, 210], 'height': [152, 154, 156, 158, 160]},
    'LAX': {'weight': [2600, 2700, 2800, 2900, 3000], 'volume': [130, 132, 134, 136, 138], 'length': [340, 350, 360, 370, 380], 'width': [220, 222, 224, 226, 228], 'height': [170, 172, 174, 176, 178]},
    'SFO': {'weight': [2700, 2800, 2900, 3000, 3100], 'volume': [135, 137, 139, 141, 143], 'length': [350, 360, 370, 380, 390], 'width': [225, 227, 229, 231, 233], 'height': [175, 177, 179, 181, 183]},
    'YVR': {'weight': [2800, 2900, 3000, 3100, 3200], 'volume': [140, 142, 144, 146, 148], 'length': [360, 370, 380, 390, 400], 'width': [230, 232, 234, 236, 238], 'height': [180, 182, 184, 186, 188]},
    'BKK': {'weight': [1850, 1950, 2050, 2150, 2250], 'volume': [92, 94, 96, 98, 100], 'length': [265, 275, 285, 295, 305], 'width': [182, 184, 186, 188, 190], 'height': [132, 134, 136, 138, 140]},
    'MAD': {'weight': [2250, 2350, 2450, 2550, 2650], 'volume': [118, 120, 122, 124, 126], 'length': [315, 325, 335, 345, 355], 'width': [208, 210, 212, 214, 216], ' height': [158, 160, 162, 164, 166]},
    'HGH': {'weight': [1550, 1650, 1750, 1850, 1950], 'volume': [76, 78, 80, 82, 84], 'length': [235, 245, 255, 265, 275], 'width': [175, 177, 179, 181, 183], 'height': [125, 127, 129, 131, 133]},
    'DAT': {'weight': [1350, 1450, 1550, 1650, 1750], 'volume': [66, 68, 70, 72, 74], 'length': [210, 220, 230, 240, 250], 'width': [157, 159, 161, 163, 165], 'height': [107, 109, 111, 113, 115]},
    'CHG': {'weight': [1420, 1520, 1620, 1720, 1820], 'volume': [64, 66, 68, 70, 72], 'length': [218, 228, 238, 248, 258], 'width': [156, 158, 160, 162, 164], 'height': [106, 108, 110, 112, 114]},
    'DPS': {'weight': [1050, 1150, 1250, 1350, 1450], 'volume': [58, 60, 62, 64, 66], 'length': [190, 200, 210, 220, 230], 'width': [148, 150, 152, 154, 156], 'height': [98, 100, 102, 104, 106]},
    'SYD': {'weight': [2900, 3000, 3100, 3200, 3300], 'volume': [145, 147, 149, 151, 153], 'length': [370, 380, 390, 400, 410], 'width': [235, 237, 239, 241, 243], 'height': [185, 187, 189, 191, 193]}
}

def generate_data(city, days=5):
    data = city_data.get(city, {
        'weight': [1000 for _ in range(days)],
        'volume': [50 for _ in range(days)],
        'length': [200 for _ in range(days)],
        'width': [150 for _ in range(days)],
        'height': [100 for _ in range(days)]
    })
    return {key: values[:days] for key, values in data.items()}

@app.route('/')
def index():
    """主地图页面"""
    # 创建地图对象
    m = folium.Map(location=[20, 0], tiles='CartoDB positron', zoom_start=2, attr='CartoDB')

    # 添加城市标记
    for city in cities:
        # 自定义图标，包含蓝色图标样式、三位字母缩写和 avg 数字
        custom_icon = folium.DivIcon(
            icon_size=(80,80),
            icon_anchor=(40, 80),
            html=f'<div style="text-align: center; width: 80px; height: 80px;"><i class="fa fa-map-marker" style="font-size:15px;color:grey;"></i><br><b><span style="font-size:12pt;color:blue;">{city["name"]}</span></b><br><b><span style="font-size:10pt;color:black;">Avg: {city["avg"]}</span></b></div>'
        )
        popup_content = f'<b>{city["name"]}</b><br>'
        folium.Marker(
            location=[city['lat'], city['lon']],
            popup=folium.Popup(popup_content, max_width=250),
            icon=custom_icon
        ).add_to(m)

    # 将地图渲染为 HTML
    map_html = m._repr_html_()

    # 渲染模板并传递地图 HTML
    return render_template('index.html', map=map_html)

@app.route('/get_chart', methods=['POST'])
def get_chart():
    city = request.form.get('city')
    aircraft_type = request.form.get('aircraft_type')
    days = list(range(1, 6))  # 假设数据长度为 5，按实际需求修改
    data = generate_data(city)  # 传入城市参数获取对应数据

    plt.figure(figsize=(10, 6))
    if aircraft_type == 'widebody':
        plt.plot(days, data['weight'], label='Weight(kg)')
        plt.plot(days, data['volume'], label='Volume (m³)')
        plt.plot(days, data['length'], label='Length(cm)')
        plt.plot(days, data['width'], label='Width(cm)')
        plt.plot(days, data['height'], label='Height(cm)')
    else:
        plt.plot(days, data['weight'], label='Weight(kg)')
        plt.plot(days, data['volume'], label='Volume (m³)')

    plt.xlabel('Days')
    plt.ylabel('Values')
    plt.title(f'{city} - {aircraft_type} Transport Data')
    plt.legend()
    plt.grid(True)

    # 转换图表为 base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    return jsonify({'image_base64': image_base64})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
