// Convert a song from "tex" flat score format, to a binary "gp" format.
// Takes two arguments, e.g. `node tex_to_gtp.js score.tex score.gp`

var alphatab = require("@coderline/alphatab");
var fs = require('fs');

var args = process.argv.slice(2);
const inPath = args[0];
const outPath = args[1];

var buffer = fs.readFileSync(inPath, 'utf8');
var tex = buffer.toString();

var importer = new alphatab.importer.AlphaTexImporter();
importer.initFromString(tex, new alphatab.Settings());
var score = importer.readScore();

// var texExporter = new alphatab.exporter.AlphaTexExporter();
// texExporter.Export(score);
// console.log(tex);
// console.log();
// console.log(texExporter.ToTex());

var exporter = new alphatab.exporter.Gp7Exporter();
var outBuf = exporter.export(score);
fs.writeFileSync(outPath, outBuf);


