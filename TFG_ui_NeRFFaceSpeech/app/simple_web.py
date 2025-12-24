"""
前端服务器 - 提供前端主页和静态文件服务
运行方式: python simple_web.py
访问地址: http://localhost:7860/
"""
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import os
import sys

# 获取当前文件所在目录（gradio_app）
CURRENT_DIR = Path(__file__).parent.resolve()
WEBUI_DIR = CURRENT_DIR / "webui"

class CustomHTTPRequestHandler(SimpleHTTPRequestHandler):
    """自定义HTTP请求处理器，将根路径重定向到start.html"""
    
    def __init__(self, *args, **kwargs):
        # 设置服务目录为webui目录
        super().__init__(*args, directory=str(WEBUI_DIR), **kwargs)
    
    def end_headers(self):
        # 添加CORS头，允许跨域访问
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        # 禁用缓存，确保浏览器总是获取最新文件（开发环境）
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()
    
    def do_OPTIONS(self):
        # 处理OPTIONS请求（CORS预检）
        self.send_response(200)
        self.end_headers()
    
    def do_GET(self):
        # 如果请求根路径，返回start.html（开始页面）
        if self.path == '/' or self.path == '':
            self.path = '/start.html'
        
        # 调用父类方法处理请求
        return super().do_GET()
    
    def log_message(self, format, *args):
        """自定义日志格式"""
        # 简化日志输出
        sys.stderr.write("%s - - [%s] %s\n" %
                        (self.address_string(),
                         self.log_date_time_string(),
                         format%args))

def run_server(port=7860):
    """启动HTTP服务器"""
    server_address = ('0.0.0.0', port)
    httpd = HTTPServer(server_address, CustomHTTPRequestHandler)
    
    print("=" * 60)
    print("前端服务器已启动")
    print(f"访问地址: http://localhost:{port}/")
    print(f"静态文件目录: {WEBUI_DIR}")
    print("=" * 60)
    print("功能页面:")
    print(f"  - 开始页面: http://localhost:{port}/start.html")
    print("=" * 60)
    print("提示: 确保后端已启动（uvicorn backend.main:app --host 0.0.0.0 --port 8000）")
    print("后端地址: http://localhost:8000/")
    print("按 Ctrl+C 停止服务器")
    print("=" * 60)
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n服务器已停止")
        httpd.shutdown()

if __name__ == "__main__":
    # 检查webui目录是否存在
    if not WEBUI_DIR.exists():
        print(f"错误: webui目录不存在: {WEBUI_DIR}")
        sys.exit(1)
    
    # 检查start.html是否存在
    start_file = WEBUI_DIR / "start.html"
    if not start_file.exists():
        print(f"警告: start.html不存在: {start_file}")
        print("服务器仍会启动，但可能无法正常显示开始页面")
    
    # 启动服务器
    run_server()

