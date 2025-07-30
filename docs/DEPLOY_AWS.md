# Project Hyperion: AWS EC2 Deployment Guide (Free Tier)

## 1. Launch a Free EC2 Instance
- Go to AWS EC2 Console
- Launch an Ubuntu 22.04 (or Amazon Linux 2) t2.micro or t3.micro instance (free tier eligible)
- Allow inbound ports: 22 (SSH), 8501 (Streamlit dashboard), 80/443 (optional for web)

## 2. Connect via SSH
```
ssh -i your-key.pem ubuntu@your-ec2-public-ip
```

## 3. Install Docker & Git
```
sudo apt update && sudo apt install -y docker.io git
sudo usermod -aG docker $USER
newgrp docker
```

## 4. Clone Your Repo
```
git clone <your-repo-url>
cd project_hyperion
```

## 5. Build and Run the Docker Container
```
docker build -t hyperion .
docker run -it -p 8501:8501 hyperion
```

## 6. Run the Bot or Dashboard
- To run the bot:
  ```
  python ultra_main.py
  ```
- To run the dashboard:
  ```
  streamlit run modules/dashboard.py --server.port 8501 --server.address 0.0.0.0
  ```
- Access the dashboard at: `http://<your-ec2-public-ip>:8501`

## 7. Notes
- All dependencies are free/open-source. No paid APIs or subscriptions required.
- For persistent/production use, consider using a t3.small or larger instance.
- For security, restrict inbound ports and use SSH keys.

---
**You are now running Project Hyperion on the AWS free tier!** 