#!/usr/bin/env python3
"""Render complete bounded-experiment summaries from retained JSON evidence."""
import json
from pathlib import Path


def cell(stat):
    return f"{stat['median']:.3f} ({stat['min']:.3f}–{stat['max']:.3f})"


def main():
    here = Path(__file__).resolve().parent
    verdicts = {'benefit': '可辨识收益', 'regression': '可辨识回退', 'uncertain': '不确定'}
    lines = ['# 正式性能测量汇总', '',
             '中位数（最小—最大）；秒为基准内部正式批次计时，不是进程耗时。',
             '每次原始输出、运行次序、命令、二进制 SHA-256 和失败状态保存在对应 JSON。', '']
    for path in sorted((here / 'measurements').glob('*.json')):
        data = json.loads(path.read_text(encoding='utf-8'))
        if 'summary' not in data:
            lines.extend([f"## {data['comparison']}", '', f'未完成；原始记录：`measurements/{path.name}`。', ''])
            continue
        lines.extend([f"## {data['comparison']}", '',
                      f"结论：**{verdicts[data['summary']['verdict']]}**。"
                      f"前三对结论：{verdicts[data['initial_three_pair_verdict']]}。",
                      f'证据：`measurements/{path.name}`。', '',
                      '| 首种子 | 版本 | 次数 | total_time 秒 | policy_time 秒 | 局/秒 | 决策/秒 |',
                      '| --- | --- | ---: | ---: | ---: | ---: | ---: |'])
        for seed, record in data['summary']['seeds'].items():
            for role, stats in record['roles'].items():
                lines.append(f"| {seed} | {role} | {len(record['pairs'])} | " + ' | '.join(
                    cell(stats[field]) for field in ('total_time_seconds', 'policy_time_seconds',
                                                    'games_per_sec', 'decisions_per_sec')) + ' |')
        lines.extend(['', '逐对变化（候选/基线 − 1，负值代表耗时减少）：', '',
                      '| 首种子 | 对序号 | total_time 变化 | policy_time 变化 |',
                      '| --- | ---: | ---: | ---: |'])
        for seed, record in data['summary']['seeds'].items():
            for pair in record['pairs']:
                lines.append(f"| {seed} | {pair['pair']} | {pair['total_change_percent']:+.2f}% | "
                             f"{pair['policy_change_percent']:+.2f}% |")
        lines.append('')
    (here / 'performance-summary.md').write_text('\n'.join(lines).rstrip() + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
