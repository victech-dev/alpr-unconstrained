<template>
    <v-container >
        <v-row align="center" justify="center" >
            <v-col cols=6>
                <v-row dense align="center" justify="end" >
                    <v-btn class="ma-2" tile outlined color="success" v-on:click="showHelp=true">Help</v-btn>
                    <v-btn class="ma-2" tile dark color="indigo" v-on:click="onStart" >Start</v-btn>
                    <v-btn class="ma-2" tile dark color="indigo" v-on:click="onSubmit">Submit</v-btn>
                </v-row>
            </v-col>
            <v-col cols=6>
                <v-row dense align="center" justify="start" >
                    <h4>
                        file : {{ annoData === null ? "none" : annoData.file }}, 
                        wh=({{ image === null ? 0 : image.width }}, {{ image === null ? 0 : image.height }})
                    </h4>
                </v-row>
            </v-col>

            <v-dialog
                v-model="showHelp"
                max-width="800"
                >
                <v-card>
                    <v-card-title class="headline">
                        Wpod 데이터 편집 가이드
                    </v-card-title>

                    <v-card-text>
                        <v-divider></v-divider>
                        <br>
                        <p>
                            'START' 버튼으로 이미지 한 장을 받아옵니다.<br>
                            번호판에 대해서 사각형를 라벨링 해야 합니다.<br>
                            이미 적당한 label들이 존재하고, 요 점들을 적절히 옮겨주시면 됩니다.<br>
                            <b>점의 기준은 숫자 영역을 감싸고 있는 가장 안쪽의 4각형 영역입니다.</b><br>
                            left-top이 첫번째 점(파란 점)이 되어야 하고 시계 방향 순서입니다.<br>
                            이후 'SUBMIT' 으로 제출합니다. <b>(여러번 submit 가능!!)</b>
                        </p>
                        <v-divider></v-divider>
                        <br>
                        <ul>
                            <li><b>화면 확대</b> : 마우스로 적당히 드래그앤드랍</li>
                            <li><b>화면 확대 취소</b> : '`' 키. (1 왼쪽의 backquote 키)</li>
                            <li><b>label 선택</b> : label 안쪽 영역을 클릭. 선택되면 색이 노란색으로 바뀝니다.</li>
                            <li><b>label 선택취소</b> : Esc 키 혹은 다시 클릭</li>
                            <li><b>점 옮기기</b> : 선택된 상태에서 <b>우</b>클릭하면 현재 가장 가까운 점이 클릭한 위치로 올겨집니다.</li>
                            <li><b>label 추가</b> : 미선택 상태에서 c 키를 누르고 네 점을 <b>우</b>클릭</li>
                            <li><b>label 삭제</b> : 선택 상태에서 d 키</li>
                        </ul>
                        <br>
                        <v-divider></v-divider>
                        
                    </v-card-text>

                    <v-card-actions>
                        <v-spacer></v-spacer>
                        <v-btn color="blue darken-1" text @click="showHelp=false;setFocusToCanvas()">Close</v-btn>
                    </v-card-actions>
                </v-card>
            </v-dialog>

        </v-row>

        <v-row align="start" justify="center">
            <canvas
                id="imgCanvas" 
                tabindex="0" 
                v-bind:width="width" 
                v-bind:height="height" 
                @keydown="onKeyDown" 
                @mousedown="onMouseDown" 
                @mouseup="onMouseUp"
                @click="onMouseClick"
                @contextmenu.prevent="onMouseRClick"/>
        </v-row>

        <v-snackbar
            v-model="submitResultShow"
            :color="submitResultColor"
            timeout=6000
            :bottom="true"
        >
            {{ submitResult }}
            <v-btn dark text @click="submitResultShow = false">
            Close
            </v-btn>
        </v-snackbar>
    </v-container>
</template>

<script>

function clamp(num, min, max) {
  return num <= min ? min : num >= max ? max : num;
}

function loadImage(url) {
    return new Promise(r => { 
        let i = new Image(); 
        i.onload = (() => r(i)); 
        i.src = url; 
    });
}

class Point {
    constructor(x = 0, y = 0) { this.x = x; this.y = y }
    set(x, y) { this.x = x; this.y = y; return this }
    fromXY(x, y) { return new Point(x, y) }
    move(dx, dy) { this.x += dx; this.y += dy; return this }
    copy() { return new Point(this.x, this.y) }
    sqrdist(pt) { return Math.pow(Math.abs(pt.x - this.x), 2) + Math.pow(Math.abs(pt.y - this.y), 2) }
    dist(pt) { return Math.sqrt(this.sqrdist(pt)) }
    getJson(w, h) { return {"x": clamp(this.x / w, 0.0, 1,0), "y": clamp(this.y / h, 0.0, 1.0) } }
}

class Label {
    constructor(pt0 = new Point(), pt1 = new Point(), pt2 = new Point(), pt3 = new Point()) { 
        this.pt0 = pt0; this.pt1 = pt1; this.pt2 = pt2; this.pt3 = pt3 
    }
    setByPts(pt0, pt1, pt2, pt3) { 
        this.pt0 = pt0; this.pt1 = pt1; this.pt2 = pt2; this.pt3 = pt3;
        return this
    }
    setByValues(pt0x, pt0y, pt1x, pt1y, pt2x, pt2y, pt3x, pt3y) { 
        this.pt0.set(pt0x, pt0y); this.pt1.set(pt1x, pt1y); this.pt2.set(pt2x, pt2y); this.pt3.set(pt3x, pt3y);
        return this
    }
    copy() { 
        return new Label(this.pt0.copy(), this.pt1.copy(), this.pt2.copy(), this.pt3.copy())
    }
    pts() { 
        return  [ this.pt0, this.pt1, this.pt2, this.pt3 ]
    }
    pt(i) {
        if (i == 0) return this.pt0
        else if (i == 1) return this.pt1
        else if (i == 2) return this.pt2
        else if (i == 3) return this.pt3
        else return null
    }
    getBb() {
        var l = Math.min.apply(Math, this.pts().map(function(pt) { return pt.x; }))
        var r = Math.max.apply(Math, this.pts().map(function(pt) { return pt.x; }))
        var t = Math.min.apply(Math, this.pts().map(function(pt) { return pt.y; }))
        var b = Math.max.apply(Math, this.pts().map(function(pt) { return pt.y; }))
        return [l, r, t, b]
    }
    move(dx, dy) { 
        this.pts().map(function(pt) { pt.x += dx; pt.y += dy });
        return this
    }
    fineNearest(pt) {
        // return [pt, index]
        return this.pts()
            .map((p, i) => [p, i])
            .reduce((p, cur) => (pt.sqrdist(p[0]) < pt.sqrdist(cur[0]) ? p : cur));
    }
    include(pt) {
        // https://github.com/substack/point-in-polygon/blob/master/index.js
        var l = this.pts()
        var inside = false;
        for (var i = 0, j = l.length - 1; i < l.length; j = i++) {
            var xi = l[i].x, yi = l[i].y;
            var xj = l[j].x, yj = l[j].y;
            
            var intersect = ((yi > pt.y) != (yj > pt.y))
                && (pt.x < (xj - xi) * (pt.y - yi) / (yj - yi) + xi);
            if (intersect) inside = !inside;
        }
        return inside;
    }
    getJson(w, h) {
        return [
            this.pt0.getJson(w, h),
            this.pt1.getJson(w, h),
            this.pt2.getJson(w, h),
            this.pt3.getJson(w, h)
        ]
    }
}

export default {
    created () {
        //this.onStart
    },
    mounted() {
        var c = document.getElementById("imgCanvas")
        this.canvas = c.getContext('2d')
        this.canvas.imageSmoothingEnabled = false
    },
    data () {
        return {
            annoData: [],

            canvas: null,
            width: 500, // Do not set directly, call updateView()
            height: 500, // Do not set directly, call updateView()

            image: null,
            scale: 1.0, // image view scale
            viewRect: { l: 0, t: 0, w: 500, h: 500 }, // image view rect
            
            labels: [
                //new Label().setByValues(10, 10, 15, 100, 120, 80, 90, 33).move(100, 100)
            ],
            curLabel: null,

            isMouseProcessing: false,
            mx: 0,
            my: 0,

            ignoreNextClick: false,

            isLabelCreating: false,
            creatingLabel: [],

            submitResult: "",
            submitResultShow: false,
            submitResultColor: '',

            showHelp: false,
        }
    },
    methods: {
        async handleNewData() {
            this.labels = []

            const image = await loadImage(this.annoData.file)
            this.image = image
            this.createLabels()
            this.setViewRect(0, 0, image.width, image.height)
            this.updateView()
            this.setFocusToCanvas()
        },

        createLabels() {
            let w = this.image.width
            let h = this.image.height
            for (let label of this.annoData.labels) {
                if (label.length != 4) {
                    continue
                }

                let pts = []
                for (let i = 0; i < label.length; ++i) {
                    pts.push(new Point(label[i].x * w, label[i].y * h))
                }
                this.labels.push(new Label(pts[0], pts[1], pts[2], pts[3]))
            }
        },

        onStart() {
            var vm = this
            this.$http.get('/wpodannos')
                .then(function(response) {
                    vm.annoData = response.data
                    vm.scale = 1.0
                    vm.curLabel = null
                    vm.handleNewData()
                })
        },

        getLabelsJson() {
            let labelsJson = []
            for (let label of this.labels) {
                labelsJson.push(label.getJson(this.image.width, this.image.height))
            }
            return labelsJson
        },

        submit() {
            if (this.isLabelCreating === true ||
                this.isMouseProcessing === true) {
                alert("Cannot now!")
                return
            }

            if (this.labels.length === 0) {
                alert("No label data!")
                return
            }

            let postData = {
                "file": this.annoData.file,
                "labels": this.getLabelsJson()
            }

            var vm = this
            this.$http.post('/wpodannos', postData)
                .then(function (response) {
                    vm.showSubmitResult(response.status)
                })
                .catch(function (error) {
                    vm.showSubmitResult(error.response.status)
                });
        },

        onSubmit() {
            this.submit()
            this.setFocusToCanvas()
        },

        showSubmitResult(code) {
            if (code === 200) {
                this.submitResultColor = 'success'
                this.submitResult = "OK ( updated again!)"
                this.submitResultShow = true
            }
            else if (code === 201) {
                this.submitResultColor = 'success'
                this.submitResult = "OK ( created )"
                this.submitResultShow = true
            }
            else if (code === 404) {
                this.submitResultColor = 'error'
                this.submitResult = "Fail (Already submitted)"
                this.submitResultShow = true
            }
            else {
                this.submitResultColor = 'error'
                this.submitResult = "Fail (code=" + code.toString() + ")"
                this.submitResultShow = true
            }
        },

        // canvas coordinate -> image coordinate
        toImageCoord(vx, vy) {
            var vr = this.viewRect;
            var ix = vx / this.width * vr.w + vr.l
            var iy = vy / this.height * vr.h + vr.t
            return new Point(ix, iy)
        },
        // image coordinate -> canvas coordinate
        fromImageCoord(ix, iy) {
            var vr = this.viewRect;
            var vx = (ix - vr.l) * this.width / vr.w
            var vy = (iy - vr.t) * this.height / vr.h
            return new Point(vx, vy)
        },

        onMouseDown(e) {
            //console.log("onMouseDown", e)

            this.mx = e.offsetX;
            this.my = e.offsetY;
            this.isMouseProcessing = true;
        },

        onMouseUp(e) {
            //console.log("onMouseUp")

            if (this.isMouseProcessing === true) {
                this.isMouseProcessing = false
                
                if (Math.abs(this.mx - e.offsetX) >= 10 && Math.abs(this.my - e.offsetY) >= 10) {
                    var l = Math.min(this.mx, e.offsetX)
                    var r = Math.max(this.mx, e.offsetX)
                    var t = Math.min(this.my, e.offsetY)
                    var b = Math.max(this.my, e.offsetY)
                    var pt0 = this.toImageCoord(l, t)
                    var pt1 = this.toImageCoord(r, b)
                    this.setViewRect(pt0.x, pt0.y, pt1.x - pt0.x, pt1.y - pt0.y)
                    this.scale = this.image.width / (pt1.x - pt0.x)
                    this.updateView()

                    this.ignoreNextClick = true
                }
            }

            // TODO click event 무시
        },

        onMouseClick(e) {
            //console.log("onMouseClick", e)

            if (this.ignoreNextClick) {
                this.ignoreNextClick = false
                return
            }

            var pt = this.toImageCoord(e.offsetX, e.offsetY)

            if (this.isLabelCreating === true) {
                return
            }

            if (this.curLabel != null) {
                this.curLabel = null
                this.updateView()
                return
            }
            else if (this.labels.length > 0) {
                // select including label
                this.curLabel = null
                for (var i = 0; i < this.labels.length; i++) {
                    if (this.labels[i].include(pt) == true) {
                        this.curLabel = this.labels[i]
                        break
                    }
                }
                this.updateView()
                return
            }
        },

        onMouseRClick(e) {
            //console.log("onMouseLClick", e)

            var pt = this.toImageCoord(e.offsetX, e.offsetY)

            if (this.isLabelCreating === true) {
                var l = this.creatingLabel
                l.push(pt)
                if (this.creatingLabel.length == 4) {
                    this.labels.push(new Label(l[0], l[1], l[2], l[3]))
                    this.isLabelCreating = false
                    this.creatingLabel = []
                }
                this.updateView()
                return
            }

            if (this.curLabel != null) {
                var nearest = this.curLabel.fineNearest(pt)
                if (nearest[0].dist(pt) < 100) {
                    nearest[0].set(pt.x, pt.y)
                    this.updateView()
                }
                return
            }
        },

        onKeyDown(e) {
            //console.log(e)

            if (this.isMouseProcessng === true) {
                if (e.key == 'Escape') {
                    if (this.isMouseProcessing) {
                        this.isMouseProcessing = false
                    }
                }
            }

            if (this.isLabelCreating === true) {
                if (e.key == 'Escape') {
                    this.isLabelCreating = false
                    this.creatingLabel = []
                    this.updateView()
                }
            }

            if (this.curLabel != null) {
                if (e.key == 'Escape') {
                    this.curLabel = null
                    this.updateView()
                }
            }

            if (e.key == 's') {
                if (e.ctrlKey == true) {
                    this.onSubmit()
                    e.preventDefault();
                    return
                }
            }
            if (e.key == 'n') {
                if (e.ctrlKey == true) {
                    this.onStart()
                    e.preventDefault();
                    return
                }
            }
            if (e.key == '`') {
                // reset
                this.scale = 1
                this.setViewRect(0, 0, this.image.width, this.image.height)
                this.updateView()
                return
            }
            if (e.key == 'c') {
                // label creating
                this.isLabelCreating = true
                this.creatingLabel = []
                return
            }
            if (e.key == 'd') {
                // delete selected label
                if (this.curLabel != null) {
                    var idx = this.labels.indexOf(this.curLabel)
                    if (idx >= 0) {
                        this.labels.splice(idx, 1)
                        this.updateView()
                    }
                    return
                }
            }
        },

        drawLabel(pts, lineColor="red", ptColor="blue") {
            if (pts.length == 4) {
                pts.push(pts[0].copy())
            }

            var vm = this
            pts = pts.map(function(pt) { 
                return vm.fromImageCoord(pt.x, pt.y)
            })

            var ctx = this.canvas

            // draw label
            ctx.lineWidth = "1";
            ctx.strokeStyle = lineColor
            ctx.beginPath()
            ctx.moveTo(pts[0].x, pts[0].y)
            for (var i = 1; i < pts.length; i++) {
                ctx.lineTo(pts[i].x, pts[i].y)
            }
            ctx.stroke()

            // draw first point
            ctx.fillStyle = ptColor;
            ctx.fillRect(pts[0].x - 2, pts[0].y - 2, 4, 4);
        },

        updateView() {
            if (this.canvas === null || this.image === null) {
                return
            }

            let vr = this.viewRect
            this.width = vr.w * this.scale
            this.height = vr.h * this.scale

            this.$nextTick(() => {
                this.canvas.drawImage(this.image,
                    vr.l, vr.t, vr.w, vr.h,
                    0, 0, this.width, this.height)
                
                for (let label of this.labels) {
                    var labelLineColor = (label === this.curLabel ? "yellow" : "red")
                    this.drawLabel(label.pts(), labelLineColor)
                }

                if (this.isLabelCreating) {
                    this.drawLabel(this.creatingLabel)
                }
            })
        },

        setViewRect(l, t, w, h) {
            this.viewRect.l = l;
            this.viewRect.t = t;
            this.viewRect.w = w;
            this.viewRect.h = h;
        },

        setFocusToCanvas() {
            document.getElementById("imgCanvas").focus()
        },
    },
}
</script>

<style scoped>

#imgCanvas {
    border: 1px solid grey;
}

</style>

