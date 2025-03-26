# You may need to update this with new RunPod instances
PORT ?= 46426
HOST ?= root@195.26.232.177
SSH_KEY ?= ~/.ssh/runpod

deploy:
	rsync -avzr --no-owner --no-group --exclude="*.git" --exclude="*.DS_Store" . runpod:/workspace/SkyThought

ssh:
	ssh -i $(SSH_KEY) -p $(PORT) $(HOST)

scp_setup_runpod:
	scp -i $(SSH_KEY) -P $(PORT) scripts/setup_runpod.yml $(HOST):/workspace
	ssh -i $(SSH_KEY) -p $(PORT) $(HOST) "pip install ansible"
	@echo "Playbook copied successfully to /workspace/SkyThought/scripts/setup_runpod.yml"
	@echo "Next steps:"
	@echo "ssh -i $(SSH_KEY) -p $(PORT) $(HOST)"
	@echo "ansible-playbook /workspace/SkyThought/scripts/setup_runpod.yml"
