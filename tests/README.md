# Tests

离线测试分两层，都可从仓库根目录运行：

```bash
python3 -m unittest discover -s tests -v   # 标准 unittest（stdlib）
python3 -m tests.run_offline               # 21 个离线套件
python3 -m tests.run_offline t_claims_unit # 只跑指定套件
```

`tests/offline/` 保留原套件的脚本式断言；`tests/run_offline.py` 逐个子进程
运行并把工作目录固定在仓库根目录。前端套件需要本机已安装 `esprima` 与
`quickjs`，未安装时这些套件会失败，其余 stdlib 套件不受影响。

`tests/fixtures/` 保存测试所需的稳定数据：

- `history/*.json`：10 份已脱敏 `owner` 的综述快照，用于逐句校验、徽标和
  参考文献折叠回归。
- `graph_af2.json`、`pool_graph_real.json`：图谱布局与整轮全池图谱的真实
  响应 fixture。

测试不访问网络、不绑定端口，也不依赖 `/tmp/sm` 或运行时的 `outputs/`。
