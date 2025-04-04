# You may need to update this with new RunPod instances

deploy:
	rsync -avzr --no-owner --no-group --exclude="*.git" --exclude="*.DS_Store" . runpod:/workspace/SkyThought

scp_setup_runpod:
	scp scripts/setup_runpod.yml runpod:/workspace
	mkdir -p runpod:/workspace/SkyThought
	scp .env runpod:/workspace/SkyThought
	ssh runpod "pip install ansible"
	@echo "Playbook copied successfully to /workspace/setup_runpod.yml"
	@echo "Next steps:"
	@echo "ssh runpod"
	@echo "ansible-playbook /workspace/setup_runpod.yml"
