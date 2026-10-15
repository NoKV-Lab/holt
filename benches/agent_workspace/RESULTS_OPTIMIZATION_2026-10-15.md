# DB::view 捕获与 COW 回收优化：最终复测

本次改动解决跨 Tree view 的重复捕获工作，以及短命 COW 版本不断进入
checkpoint/store 的放大。没有改变公开 API、磁盘格式、跨 Tree 一致性、
escaped view/cursor 生命周期或 WAL 同步确认边界。

基线：`f080daf2bc9d8019bb134141dc5ee31fd36f2c03`（0.9.2）。
修改版：测量时为同一基线上的本地补丁（`git diff --binary`），补丁 SHA256
`53a9b55f9a0dd8df8763229d1892c46863b1463680c40b4ffd222f29412157f1`。
两版使用完全相同的 Rust workspace handler、Cargo.lock 和 release 设置。
SQLite 为同一驱动中的 bundled 3.51.3。

提交 PR 时另整合了 main 的 `36fb295`（manifest dual-writer 防护）及依赖更新；
下列正式性能数字仍对应上述测量版本，不冒充整合后 main 的完整复测结果。

## 测量边界

Apple M3 Pro、36 GiB RAM、12 logical CPUs，macOS，Rust 1.97.1。
8 workspaces × 512 预置 paths；原有八类 metadata、请求比例、JSON 编解码、
guarded mutation、内容 CRC、4 KiB 内容读写和 checkpoint 策略均未变。
这不是 NoKV SDK/RPC 或远端对象存储测试。

11 个场景，原版 / 修改版 / SQLite 各 4 次独立进程，共 132 次；另为
attached metadata 同步档补测 12 次。采用正反配对顺序，保留每次测量、
全部延迟样本、环境/磁盘余量、峰值 RSS、命令和二进制校验值。
144 次均通过完整有序域指纹（CRC32、各表行数/字节数）及重开校验；CAS 重试为 0。
测量时没有并行运行编译器、测试或其他本任务基准。桌面背景负载仍然存在。

以下吞吐不含最后一次 checkpoint；完整报告同时给出含最终 drain 的吞吐。
异步 OFF 是诊断档，不能当作 Holt/SQLite 崩溃语义等价的生产对比。
`background_normal` 保留各自原生策略：Holt 500 ms / 512 dirty blobs，
SQLite NORMAL / 1000-page autocheckpoint，并非相同刷盘节奏。同步档使用
Holt sync=true、SQLite FULL，并打开 macOS fullfsync/checkpoint_fullfsync。
未额外调用 vacuum；后台维护保持原配置。

## 请求吞吐：每组进程中位数

| 场景 | 原版 Holt ops/s | 修改版 Holt ops/s | SQLite ops/s | 修改版/原版 |
|---|---:|---:|---:|---:|
| metadata async，1 client，12k ops | 19,674 | 23,188 | 14,952 | 1.18× |
| metadata async，4 clients，4k ops，无后台 | 1,003 | 7,102 | 15,133 | 7.08× |
| metadata async，4 clients，12k ops，无后台 | 226 | 5,826 | 13,725 | 25.76× |
| metadata async，4 clients，4k ops，原生后台/NORMAL | 1,971 | 22,098 | 1,877 | 11.21× |
| content async，1 client，6k ops | 2,217 | 2,190 | 1,707 | 0.99× |
| content async，4 clients，6k ops，无后台 | 761 | 4,593 | 4,128 | 6.03× |
| content async，4 clients，4k ops，原生后台/NORMAL | 1,746 | 3,596 | 1,860 | 2.06× |
| metadata sync，1 client，4k ops | 774 | 772 | 793 | 1.00× |
| content sync，1 client，1.6k ops | 390 | 402 | 351 | 1.03× |
| attached metadata sync，1 client，4k ops | 804 | 780 | 838 | 0.97× |
| attached content sync，1 client，1.6k ops | 401 | 409 | 340 | 1.02× |

新版本四并发 metadata 有明显运行波动：4k 无后台为 6,425–21,126 ops/s，
12k 无后台为 5,586–12,546 ops/s，不能只引用最快一次。
短后台负载可能在首个周期 checkpoint 前结束，因此 11.21× 不代表长程稳态。
另外完成的 100k 后台 metadata 正确性压力测试约为 3,523 ops/s；该次与测试
并行运行，不纳入正式性能矩阵，也不是严格稳态资格测试。

## 放大与最终 drain

| 场景 | 峰值 RSS MiB，旧 → 新 | metadata allocated MiB，旧 → 新 | 最终 checkpoint ms，旧 → 新 | checkpoint 后 COW backlog，旧 → 新 |
|---|---:|---:|---:|---:|
| metadata 4 clients / 4k / 无后台 | 1,388 → 87.6 | 1,325 → 21.2 | 1,561 → 40.3 | 1,068.5 → 0 |
| metadata 4 clients / 12k / 无后台 | 6,800 → 107.0 | 8,160 → 40.7 | 12,620 → 70.1 | 7,849.5 → 0 |
| metadata 4 clients / 原生后台 | 423 → 79.9 | 1,430 → 26.2 | 159.7 → 38.0 | 897.5 → 0 |
| content 4 clients / 无后台 | 1,824 → 75.4 | 1,774 → 28.7 | 2,033 → 54.8 | 1,498 → 0 |
| content 4 clients / 原生后台 | 406 → 72.3 | 1,111 → 27.4 | 32.3 → 39.6 | 572 → 0 |

12k metadata 包含最后 drain 的吞吐为 183 → 5,644 ops/s（30.9×）；
请求 p95 为 61.3 ms → 2.61 ms。Holt 的 64 MiB 仍只是 cache target，
不是严格 RSS 上限；SQLite 实测 RSS 更低，不能声称两者总内存相等。

单线程 metadata read p50：38.54 → 29.83 µs；stat：38.63 → 29.98 µs；
8-family snapshot：154.10 → 134.69 µs。view+reads 阶段总计
501.9 → 411.4 ms；这是包含实际查询的阶段计时，不是纯 memcpy/capture 微基准。

## 同步档的补充确认

attached metadata 首轮 804.2 → 779.9 ops/s（-3.0%），补测
796.9 → 795.0 ops/s（-0.24%）。首轮新旧范围为 773–845 / 774–809，
补测为 777–818 / 758–806，均重叠。两轮每个 Holt 进程 WAL sync 数均为
1,492；view+reads 阶段均下降，apply/fsync 阶段有波动。
没有复现稳定的同步退化，也不能据此承诺所有负载绝对零回退。

## 实现与验证要点

- 多根复制后批量注册独立 epoch lease；完整覆盖的 buffer 不再先清零。
  named-tree 捕获不再逐次脏写根，DB checkpoint 在 catalog 根汇总 epoch。
- orphan 仅等待创建至脱离之间真正可能引用它的 epoch；按阻塞 epoch 索引，
  后来的快照不能延长旧版本寿命。GC 根 pin 也保留 epoch 保护。
- 未进入底层 store 的退役版本可被有界丢弃；新副本/合并父页继承 WAL 低水位。
  子页在释放写 latch 前发布 dirty debt。已持久化 GUID 仍等待安全回收边界。
- dirty capture 与临时版本回收互斥；只有旧 flushing claims 全部退出，才整体
  取消退役图像，避免旧 checkpoint 父图像依赖已丢弃的子图像。
  回收队列非空时不再每次 view drop 都唤醒后台线程。

迭代先导测试确实抓到上述父子依赖竞态，相关失败日志保留但不计入正式结果。
最终版本通过 484 个 all-features 单元测试、整套 lib/integration/example 测试、
Clippy、doctest，以及额外的 No-WAL 并发快照/写入/checkpoint 测试。
最终源码还通过 100k 操作的 DB oracle soak，以及 30 秒、95 轮
SIGKILL/reopen 的已确认跨 Tree 写入恢复 smoke（非长期资格测试）。
新增进程退出测试验证多代 COW 被丢弃后、不做最终 checkpoint，sync WAL
仍能恢复全部已确认数据。另完成 100k metadata、40k 不同 seed metadata、
12k content 的后台正确性压力测试，均重开成功且无后台错误日志。
40k 扩展测试在一次有界清理后仍有 76 个待回收 GUID，不是无限积压已被全面
证明消失；本轮也没有做 Linux/io_uring、断电或长时间稳态资格测试。

仍有每个 Tree 一次 512 KiB 根帧复制，以及全局 COW/GC epoch 的保守开销。
SQLite OFF 在四并发 metadata 的吞吐和 p95 上仍领先；本次不是所有维度的
SQLite 胜负结论，也未修改 NoKV 的 OCC、业务历史或 outbox 保留策略。

聚合数据见 `results/optimization-2026-10-15/`。原始证据与精确源码包保留在本地
`results/optimization-evidence-2026-10-15.tar.gz`，不提交大体积原始日志及桌面进程
环境快照；SHA256 与内容说明见 `results/README.md`。可通过同一驱动重新生成。
