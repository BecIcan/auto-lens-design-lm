# Web demo

网页只接收设计指标，只返回光路图、点列图和少量追迹指标。模型、权重、处方、候选结构和运行路径不会进入浏览器。

## 本机运行

```powershell
pip install -e ".[web]"
$env:EADLD_BACKEND_FACTORY="private_seed.runtime:create_backend"
$env:EADLD_BACKEND_CONFIG="D:\private\seed.json"
$env:EADLD_ACCESS_TOKEN="使用随机生成的长访问码"
$env:EADLD_PUBLIC_ORIGIN="http://127.0.0.1:8000"
eadld-web
```

打开 `http://127.0.0.1:8000`。

## 上线要求

- 服务仅监听 `127.0.0.1`，公网入口使用 HTTPS 反向代理。
- 私有包、权重和配置放在仓库外，只给服务账户只读权限。
- 设置唯一的 `EADLD_ACCESS_TOKEN`，不要写进仓库或网址；泄露后立即轮换。
- 设置准确的 `EADLD_PUBLIC_ORIGIN` 和 `EADLD_ALLOWED_HOSTS`。
- 反向代理覆盖客户端传入的转发头后，才设置 `EADLD_TRUST_PROXY_HEADERS=1`。
- 只运行一个 Web worker。模型并发由服务内部限制，横向扩展时改用共享限流和任务队列。
- 定期更新系统、Python 依赖和 TLS 配置；公网仅开放 443。

[`deploy/nginx.seed.conf`](../deploy/nginx.seed.conf) 给出了反向代理边界。证书路径和域名必须按服务器实际值填写。
