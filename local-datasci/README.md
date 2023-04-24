```
docker build -t datasci .
docker create --name jupyter-misc datasci
docker start <container id>
```

Vscode CTRL+SHIFT+P -> Remote-Containers: Attach to Running Container

### VSCode extension list to JSON array

```
cat vscode-extensions.txt | jq -R -s -c 'split("\n")[:-1]'
```