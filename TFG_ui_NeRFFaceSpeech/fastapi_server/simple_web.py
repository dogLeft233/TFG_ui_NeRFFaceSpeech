"""
前端服务器 - 提供前端主页和静态文件服务
运行方式: python simple_web.py
访问地址: http://localhost:7860/
"""
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import os
import sys

# 获取当前文件所在目录（fastapi_server）
CURRENT_DIR = Path(__file__).parent.resolve()
WEBUI_DIR = CURRENT_DIR / "webui"

class CustomHTTPRequestHandler(SimpleHTTPRequestHandler):
    """自定义HTTP请求处理器，将根路径重定向到index.html"""
    
    def __init__(self, *args, **kwargs):
        # 设置服务目录为webui目录
        super().__init__(*args, directory=str(WEBUI_DIR), **kwargs)
    
    def end_headers(self):
        # 添加CORS头，允许跨域访问
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_OPTIONS(self):
        # 处理OPTIONS请求（CORS预检）
        self.send_response(200)
        self.end_headers()
    
    def do_GET(self):
        # 如果请求根路径，返回start.html（选择页面）
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
    print(f"  - 选择页面: http://localhost:{port}/start.html")
    print(f"  - 前端主页: http://localhost:{port}/index.html")
    print(f"  - 视频生成: http://localhost:{port}/generate.html")
    print(f"  - 人机对话: http://localhost:{port}/chat.html")
    print(f"  - 训练模型: http://localhost:{port}/train.html")
    print("=" * 60)
    print("提示: 确保后端已启动（uvicorn main:app）")
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
    
    # 检查index.html是否存在
    index_file = WEBUI_DIR / "index.html"
    if not index_file.exists():
        print(f"警告: index.html不存在: {index_file}")
        print("服务器仍会启动，但可能无法正常显示主页")
    
    # 启动服务器
    run_server()
