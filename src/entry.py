import argparse
import uvicorn
from api import App

parser = argparse.ArgumentParser()

parser.add_argument(
    "-db",
    "--database-endpoint",
    type=str,
    required=False,
    default="http://192.168.11.129:8001",
    help="the endpoint for the database",
)
parser.add_argument(
    "-m",
    "--model-server",
    type=str,
    required=False,
    default="http://192.168.11.145:8005",
    help="the endpoint for the model generation server",
)
parser.add_argument(
    "--use-sd",
    action="store_true",
    help="use stable diffusion model server",
)
parser.add_argument(
    "-p",
    "--port",
    type=int,
    required=False,
    default=8004,
    help="port to run the server (default: 8004)",
)
parser.add_argument("--debug", action="store_true", help="enable debug mode")

args = parser.parse_args()
app = App(args.database_endpoint, args.model_server, args.debug, args.use_sd).get_app()

if __name__ == "__main__":
    uvicorn.run("entry:app", host="0.0.0.0", port=args.port, log_level="info")
