"""
data_preprocess.py

将 BAKFLGITH_LOADDATA*.csv（原始10列带表头）转换为标准15列格式（无表头）。

列映射规则：
  输入列（带表头）:  FLIGHT, TYPE, DEST, WEIGHT, FLOOR TYPE, POS, CONT, PRIORITY, VOLUME, SPECIAL CARGO
  输出列（无表头，共15列）:
    col1  <- FLIGHT          (原col1)
    col2  <- TYPE            (原col2，保持原值不变)
    col3  <- DEST            (原col3)
    col4  <- WEIGHT          (原col4)
    col5  <- FLOOR TYPE      (原col5)
    col6  <- POS             (原col6)
    col7  <- 空列            (ULD序列号，外部数据，留空)
    col8  <- CONT            (原col7，ULD型号)
    col9  <- PRIORITY        (原col8)
    col10 <- SPECIAL CARGO   (原col10)
    col11 <- 空列            (货物标签，外部数据，留空)
    col12 <- VOLUME          (原col9)
    col13 <- 飞行内序号       (同一FLIGHT的记录从1开始递增)
    col14 <- T标志           (col8为空时填"T"，否则为空)
    col15 <- 空列

用法:
    python data_preprocess.py --input-path /data/raw/ --output-path /data/processed/
"""

import sys
import os
import csv
import glob
import argparse
from collections import defaultdict


def transform_file(input_path: str, output_path: str) -> int:
    """
    读取原始CSV（10列带表头），输出标准格式CSV（15列无表头）。
    返回处理的数据行数。
    """
    # 统计每个FLIGHT已出现的行数，用于生成序号
    flight_seq = defaultdict(int)

    rows_written = 0

    # 自动检测编码（中文数据可能是 GBK/GB18030）
    for enc in ('utf-8-sig', 'utf-8', 'gbk', 'gb18030', 'latin-1'):
        try:
            with open(input_path, encoding=enc) as _f:
                _f.read()
            detected_enc = enc
            break
        except UnicodeDecodeError:
            continue
    else:
        detected_enc = 'latin-1'  # 兜底

    with open(input_path, newline='', encoding=detected_enc) as fin, \
         open(output_path, 'w', newline='', encoding='utf-8') as fout:

        reader = csv.reader(fin)
        writer = csv.writer(fout, lineterminator='\n')

        header = None
        for i, row in enumerate(reader):
            # 跳过表头行（第一行，或首列值为 'FLIGHT' 的行）
            if i == 0:
                header = row
                # 验证是否是期望的表头格式
                if len(row) >= 1 and str(row[0]).strip().upper() == 'FLIGHT':
                    continue  # 跳过表头
                # 如果第一行不是表头，按数据行处理（回退：不跳过）
                # fall through to process as data

            # 跳过空行
            if not any(cell.strip() for cell in row):
                continue

            # 确保至少有10列（不足则补空）
            while len(row) < 10:
                row.append('')

            # 原始列（0-based index）
            flight       = row[0].strip()
            type_        = row[1].strip()
            dest         = row[2].strip()
            weight       = row[3].strip()
            floor_type   = row[4].strip()
            pos          = row[5].strip()
            cont         = row[6].strip()   # CONT / ULD型号
            priority     = row[7].strip()
            volume       = row[8].strip()
            special_cargo= row[9].strip()

            # 计算飞行内序号
            flight_seq[flight] += 1
            seq = flight_seq[flight]

            # col14: T标志
            # 当 CONT/ULD型号(col8) 为空时标记为散装货物
            # 注意：若col11(货物标签)有外部赋值则应清除T标志，此处以col8判断
            t_flag = 'T' if cont == '' else ''

            # 构造输出行（15列）
            out_row = [
                flight,        # col1
                type_,         # col2
                dest,          # col3
                weight,        # col4
                floor_type,    # col5
                pos,           # col6
                '',            # col7  ULD序列号（外部数据，留空）
                cont,          # col8  CONT / ULD型号
                priority,      # col9
                special_cargo, # col10
                '',            # col11 货物标签（外部数据，留空）
                volume,        # col12
                seq,           # col13 飞行内序号
                t_flag,        # col14 T标志
                '',            # col15
            ]

            writer.writerow(out_row)
            rows_written += 1

    return rows_written



def main():
    parser = argparse.ArgumentParser(
        description='将 BAKFLGITH_LOADDATA*.csv 从原始10列格式转换为标准15列格式。'
    )
    parser.add_argument(
        '--input-path', required=True,
        help='输入目录，包含 BAKFLGITH_LOADDATA*.csv 文件'
    )
    parser.add_argument(
        '--output-path', required=True,
        help='输出目录，转换后文件与输入文件同名输出至此目录（不能与输入目录相同）'
    )
    args = parser.parse_args()

    input_dir  = os.path.abspath(args.input_path)
    output_dir = os.path.abspath(args.output_path)

    if input_dir == output_dir:
        print('[错误] --input-path 和 --output-path cannot be the same')
        sys.exit(1)

    input_files = sorted(glob.glob(os.path.join(input_dir, 'BAKFLGITH_LOADDATA*.csv')))

    if not input_files:
        print(f'[错误] 在 {input_dir} 中未找到匹配文件 BAKFLGITH_LOADDATA*.csv')
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    print(f'共找到 {len(input_files)} 个文件待处理\n')

    total_rows = 0
    for fp in input_files:
        filename = os.path.basename(fp)
        out_path = os.path.join(output_dir, filename)
        print(f'  处理: {fp}')
        print(f'  输出: {out_path}')
        try:
            n = transform_file(fp, out_path)
            print(f'  完成: {n} 行\n')
            total_rows += n
        except Exception as e:
            print(f'  [错误] {e}\n')

    print(f'全部完成，共处理 {total_rows} 行数据。')


if __name__ == '__main__':
    main()
