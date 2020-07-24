var express = require('express');
var router = express.Router();
const path = require('path');
const fs = require('fs');

/* GET users listing. */
router.get('/', function(req, res, next) {
  const dataPath = path.join(__dirname, '../data/');
  res.send(dataPath);
});

module.exports = router;
