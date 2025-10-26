import argparse
from HardwareControl.dualThreadApp import DualThreadApp

def main():
    """Entry point."""
    
    
    parser = argparse.ArgumentParser(
        description="Microscopy Experiment Control with REST API"
    )
    parser.add_argument(
        "--no-api",
        action="store_true",
        help="Disable REST API server"
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=5000,
        help="REST API server port (default: 5000)"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("MULTI-THREAD: CAMERA + STAGE + AUTOFOCUS + REST API")
    print("="*70)
    print("This will run:")
    print("  1. Camera live stream (GUI window)")
    print("  2. Stage control via CLI commands")
    print("  3. Autofocus with live plotting")
    print("  4. Interactive command prompt")
    if not args.no_api:
        print(f"  5. REST API server (port {args.api_port})")
    print("\nType 'quit' or press Ctrl+C to stop.")
    print("="*70 + "\n")
    
    app = DualThreadApp(
        enable_api=not args.no_api,
        api_port=args.api_port
    )
    app.run()
    
    print("\n" + "="*70)
    print("APPLICATION TERMINATED")
    print("="*70)


if __name__ == "__main__":
    main()