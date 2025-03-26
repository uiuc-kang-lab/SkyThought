deploy:
	rsync -avzr --no-owner --no-group --exclude="*.git" --exclude="*.DS_Store" . runpod:/workspace/SkyThought