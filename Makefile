ship:
	@mkdir -p work; \
	cd work; \
	cp ../target/release/mdidx . ;\
	cp ../target/release/mdidx-mcp . ;\
	cp ../target/release/mdidx-mcp-http . ;\
	zip -r mdidx_macos.zip mdidx mdidx-mcp mdidx-mcp-http
