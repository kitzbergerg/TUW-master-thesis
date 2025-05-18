Run `npm start` to serve the site on http://localhost:8080/.  
Run `npx serve model --cors` to serve the model files.

To generate `data.json` run

```nushell
cd model/
ls phi/torch/ | get name | each {|path| { path: ($path | path basename), data: $"http://localhost:3000/($path)" } } | to json | save -f ../public/phi/data.json
```
