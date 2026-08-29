# Web demo

网页只接收设计指标，只返回光路图、点列图、少量追迹指标和用户主动导出的 SEQ。模型、权重、候选池和运行路径不会进入浏览器。

## 本机运行

```powershell
pip install -e ".[web]"
$env:EADLD_BACKEND_FACTORY="private.seed_backend:create_backend"
$env:EADLD_PUBLIC_ORIGIN="http://127.0.0.1:8000"
$env:EADLD_DAILY_LIMIT="5"
$env:EADLD_QUOTA_SECRET="使用随机生成的长密钥"
eadld-web
```

打开 `http://127.0.0.1:8000`。

## 上线要求

- 服务仅监听 `127.0.0.1`，公网入口使用 HTTPS 反向代理。
- 私有包、权重和配置放在仓库外，只给服务账户只读权限。
- 免登录模式不设置 `EADLD_ACCESS_TOKEN`；访客按服务端识别的 IP 每日限次。
- 设置随机的 `EADLD_QUOTA_SECRET`，不要提交；额度数据库只保存访客标识摘要。
- 设置准确的 `EADLD_PUBLIC_ORIGIN` 和 `EADLD_ALLOWED_HOSTS`。
- Cloudflare Tunnel 下设置 `EADLD_CLIENT_IP_HEADER=cf-connecting-ip`。源站必须保持仅本机可达，避免伪造该请求头。
- 只运行一个 Web worker。模型并发由服务内部限制，横向扩展时改用共享限流和任务队列。
- 定期更新系统、Python 依赖和 TLS 配置；公网仅开放 443。

匿名限次适合小范围体验，但不能阻止用户更换网络。公网测试建议在入口再启用人机验证和异常流量规则。

[`deploy/nginx.seed.conf`](../deploy/nginx.seed.conf) 给出了反向代理边界。证书路径和域名必须按服务器实际值填写。
