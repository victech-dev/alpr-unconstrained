var express = require('express');
var router = express.Router();
const path = require('path');
const fs = require('fs');
const glob = require('glob')
const readline = require('readline');

const publicPath = path.join(__dirname, '../public/');
const dataPath = path.join(publicPath, 'data/');
var jpgFiles = []

function removeJpgFile(name) {
    const index = jpgFiles.indexOf(name);
    console.log(index)
    if (index > -1) {
        jpgFiles.splice(index, 1)
    }
}

function getRandomInt(min, max) { 
    return Math.floor(Math.random() * (max - min + 1)) + min
}

function readLabel(line) {
    var tokens = line.split(",")
    var numPoints = Number(tokens[0])
    var label = []
    for (let i = 0; i < numPoints; ++i) {
        var x = Number(tokens[1 + i])
        var y = Number(tokens[1 + numPoints + i])
        label.push({ "x": x, "y": y})
    }
    return label
}

async function readLabels(fileName) {
    const ifs = fs.createReadStream(fileName);
  
    const lineReader = readline.createInterface({
      input: ifs,
      crlfDelay: Infinity
    })

    labels = []
    for await (const line of lineReader) {
        labels.push(readLabel(line))
    }
    return labels
}

function writeLabel(label) {
    var line = "".concat(label.length.toString(), ",")
    for (var i = 0; i < label.length; ++i) {
        line = line.concat(label[i].x.toString(), ",")
    }
    for (var i = 0; i < label.length; ++i) {
        line = line.concat(label[i].y.toString(), ",")
    }
    line = line.concat(",")
    return line
}

function writeLabels(fileName, labels) {
    var ofs = fs.createWriteStream(fileName, {
        flags: 'w'
    })
    for (var i = 0; i < labels.length; ++i) {
        ofs.write(writeLabel(labels[i]))
    }
    ofs.end()
}

function changeExt(baseFile, ext) {
    var parsed = path.parse(baseFile)
    return parsed.dir + "/" + parsed.name + ext
}

function checkFile(baseFile, ext) {
    return fs.existsSync(changeExt(baseFile, ext))
}

function getPublicRelPath(file) {
    return file.substring(publicPath.length)
}

var options = {}
glob(dataPath + "*.jpg", options, function (er, files) {
    files.forEach(function (file) {
        if (checkFile(file, ".txt") === false) {
            if (file.startsWith(publicPath)) {
                jpgFiles.push(file)
            }
        }
    });
});

/* GET random jpg file and its labels */
router.get('/', async function(req, res, next) {
    var idx = getRandomInt(0, jpgFiles.length - 1)
    var jpgFile = jpgFiles[idx]

    var preTxtFile = changeExt(jpgFile, "_pre.txt")
    var labels = await readLabels(preTxtFile)
    res.json({ "file": getPublicRelPath(jpgFile), "labels": labels });
});

/* POST save jpg file and its labels */
router.post('/', async function(req, res, next) {
    jpgFile = path.join(publicPath, req.body.file);

    let idx = jpgFiles.indexOf(jpgFile)
    if (idx < 0) {
        res.sendStatus(404)
    } else {
        let outTxtFile = changeExt(jpgFile, ".txt")
        try {
            writeLabels(outTxtFile, req.body.labels)
            idx = jpgFiles.indexOf(jpgFile)
            if (idx > -1)
                jpgFiles.splice(idx, 1)
            res.sendStatus(200)
        }
        catch (e) {
            res.sendStatus(400)
        }
    }
});

module.exports = router;