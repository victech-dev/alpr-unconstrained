var express = require('express');
var router = express.Router();
const path = require('path');
const fs = require('fs');
const glob = require('glob')
const readline = require('readline');

const publicPath = path.join(__dirname, '../public/');
const dataPath = path.join(publicPath, 'data_ocr/');
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

function readBb(line) {
    let tokens = line.split(" ")
    let cls = Number(tokens[0])
    let cx = Number(tokens[1])
    let cy = Number(tokens[2])
    let w = Number(tokens[3])
    let h = Number(tokens[4])
    return { "cls": cls, "cx": cx, "cy": cy, "w": w, "h": h }
}

async function readBbList(fileName) {
    const ifs = fs.createReadStream(fileName);
  
    const lineReader = readline.createInterface({
      input: ifs,
      crlfDelay: Infinity
    })

    bbList = []
    for await (const line of lineReader) {
        bbList.push(readBb(line))
    }
    return bbList
}

function writeBb(bb) {
    return `${bb.cls} ${bb.cx} ${bb.cy} ${bb.w} ${bb.h}\n`
}

function writeBbList(fileName, bbList) {
    let ofs = fs.createWriteStream(fileName, {
        flags: 'w'
    })
    for (let i = 0; i < bbList.length; ++i) {
        ofs.write(writeBb(bbList[i]))
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
    //let idx = 0
    let jpgFile = jpgFiles[idx]

    console.log("** jpgFiles cnt=", jpgFiles.length)

    let preTxtFile = changeExt(jpgFile, "_pre.txt")
    let bbList = await readBbList(preTxtFile)
    res.json({ "file": getPublicRelPath(jpgFile), "bbList": bbList });
});

/* POST save jpg file and its labels */
router.post('/', async function(req, res, next) {
    jpgFile = path.join(publicPath, req.body.file);

    let idx = jpgFiles.indexOf(jpgFile)
    if (idx < 0) {
        let outTxtFile = changeExt(jpgFile, ".txt")
        if (fs.existsSync(outTxtFile)) {
            // resubmit
            writeBbList(outTxtFile, req.body.bbList)
            res.sendStatus(200)
        } else {
            res.sendStatus(404)
        }
    } else {
        let outTxtFile = changeExt(jpgFile, ".txt")
        try {
            // submit for the first time
            writeBbList(outTxtFile, req.body.bbList)
            idx = jpgFiles.indexOf(jpgFile)
            if (idx > -1)
                jpgFiles.splice(idx, 1)
            res.sendStatus(201)
        }
        catch (e) {
            res.sendStatus(400)
        }
    }
});

module.exports = router;