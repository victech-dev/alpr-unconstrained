<template>
    <div class="ocranno">
        <div>
            <button class="pure-button" @click="onStart">Start</button>--
            <button class="pure-button" @click="onSubmit">Submit</button>
        </div>
        <div>
            -- {{ submitResult }} -- 
        </div>
        
        <h5> file : {{ annoData === null ? "none" : annoData.file }}, 
            wh=({{ image === null ? 0 : image.width }}, {{ image === null ? 0 : image.height }})
        </h5>

        <div>
            <button class="pure-button" @click="onCls00">가</button>
            <button class="pure-button" @click="onCls01">나</button>
            <button class="pure-button" @click="onCls02">다</button>
            <button class="pure-button" @click="onCls03">라</button>
            <button class="pure-button" @click="onCls04">마</button>
            <button class="pure-button" @click="onCls35">바</button>
            <button class="pure-button" @click="onCls36">사</button>
            <button class="pure-button" @click="onCls37">아</button>
            <button class="pure-button" @click="onCls38">자</button>
        </div>
        <div>
            <button class="pure-button" @click="onCls05">거</button>
            <button class="pure-button" @click="onCls06">너</button>
            <button class="pure-button" @click="onCls07">더</button>
            <button class="pure-button" @click="onCls08">러</button>
            <button class="pure-button" @click="onCls09">머</button>
            <button class="pure-button" @click="onCls20">버</button>
            <button class="pure-button" @click="onCls21">서</button>
            <button class="pure-button" @click="onCls22">어</button>
            <button class="pure-button" @click="onCls23">저</button>
        </div>

        <div>
            <button class="pure-button" @click="onCls10">고</button>
            <button class="pure-button" @click="onCls11">노</button>
            <button class="pure-button" @click="onCls12">도</button>
            <button class="pure-button" @click="onCls13">로</button>
            <button class="pure-button" @click="onCls14">모</button>
            <button class="pure-button" @click="onCls24">보</button>
            <button class="pure-button" @click="onCls25">소</button>
            <button class="pure-button" @click="onCls26">오</button>
            <button class="pure-button" @click="onCls27">조</button>
        </div>
        <div>
            <button class="pure-button" @click="onCls15">구</button>
            <button class="pure-button" @click="onCls16">누</button>
            <button class="pure-button" @click="onCls17">두</button>
            <button class="pure-button" @click="onCls18">루</button>
            <button class="pure-button" @click="onCls19">무</button>
            <button class="pure-button" @click="onCls28">부</button>
            <button class="pure-button" @click="onCls29">수</button>
            <button class="pure-button" @click="onCls30">우</button>
            <button class="pure-button" @click="onCls31">주</button>
        </div>

        <div>
            <button class="pure-button" @click="onCls32">허</button>
            <button class="pure-button" @click="onCls33">하</button>
            <button class="pure-button" @click="onCls34">호</button>
            <button class="pure-button" @click="onCls39">배</button>
        </div>

        <div>
            <button class="pure-button" @click="onCls40">울</button>
            <button class="pure-button" @click="onCls41">대</button>
            <button class="pure-button" @click="onCls42">광</button>
            <button class="pure-button" @click="onCls43">산</button>
            <button class="pure-button" @click="onCls44">경</button>
            <button class="pure-button" @click="onCls45">기</button>
            <button class="pure-button" @click="onCls46">충</button>
            <button class="pure-button" @click="onCls47">북</button>
            <button class="pure-button" @click="onCls48">전</button>
            <button class="pure-button" @click="onCls49">제</button>
        </div>

        <div>
            <button class="pure-button" @click="onCls50">인</button>
            <button class="pure-button" @click="onCls51">천</button>
            <button class="pure-button" @click="onCls52">세</button>
            <button class="pure-button" @click="onCls53">종</button>
            <button class="pure-button" @click="onCls54">강</button>
            <button class="pure-button" @click="onCls55">원</button>
            <button class="pure-button" @click="onCls56">남</button>
        </div>

        <div>
            <button class="pure-button" @click="onCls57">0</button>
            <button class="pure-button" @click="onCls58">1</button>
            <button class="pure-button" @click="onCls59">2</button>
            <button class="pure-button" @click="onCls60">3</button>
            <button class="pure-button" @click="onCls61">4</button>
            <button class="pure-button" @click="onCls62">5</button>
            <button class="pure-button" @click="onCls63">6</button>
            <button class="pure-button" @click="onCls64">7</button>
            <button class="pure-button" @click="onCls65">8</button>
            <button class="pure-button" @click="onCls66">9</button>
        </div>

        <h4>
            Char : {{ curLabel === null ? "-" : getClassChar(curLabel.cls) }}
        </h4>

        <canvas id="imgCanvas" tabindex="0" 
            v-bind:width="width" v-bind:height="height" @keydown="onKeyDown"
            @click="onMouseLClick"/>
    </div>
</template>

<script>

function clamp(num, min, max) {
  return num <= min ? min : num >= max ? max : num
}

function loadImage(url) {
    return new Promise(r => { 
        let i = new Image() 
        i.onload = (() => r(i)) 
        i.src = url 
    })
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

class BbLabel {
    constructor(cls = -1, cx = 0, cy = 0, w = 0, h = 0) { 
        this.cls = cls; this.cx = cx; this.cy = cy; this.w = w; this.h = h;
    }
    copy() { 
        return new BbLabel(this.cls, this.cx, this.cy, this.w, this.h)
    }
    l() { return this.cx - this.w / 2 }
    r() { return this.cx + this.w / 2 }
    t() { return this.cy - this.h / 2 }
    b() { return this.cy + this.h / 2 }
    lset(l, limit = 0) {
        if (limit <= 0 || limit > Math.abs(this.l() - l)) {
            this.setBb(l, this.t(), this.r(), this.b())
        }
    }
    rset(r, limit = 0) { 
        if (limit <= 0 || limit > Math.abs(this.r() - r)) {
            this.setBb(this.l(), this.t(), r, this.b())
        }
    }
    tset(t, limit = 0) {
        if (limit <= 0 || limit > Math.abs(this.t() - t)) {
            this.setBb(this.l(), t, this.r(), this.b())
        }
    }
    bset(b, limit = 0) {
        if (limit <= 0 || limit > Math.abs(this.b() - b)) {
            this.setBb(this.l(), this.t(), this.r(), b)
        }
    }
    pts() { 
        // from lt, clockwise
        return  [ 
            new Point(this.l(), this.t()),
            new Point(this.r(), this.t()),
            new Point(this.r(), this.b()),
            new Point(this.l(), this.b())
        ]
    }
    pt(i) {
        if (i == 0) return new Point(this.l(), this.t())
        else if (i == 1) return new Point(this.r(), this.t())
        else if (i == 2) return new Point(this.r(), this.b())
        else if (i == 3) return new Point(this.l(), this.b())
        else return null
    }
    getBb() {
        return [this.l(), this.r(), this.t(), this.b()]
    }
    setBb(l, t, r, b) {
        if (l < r && t < b) {
            this.cx = (l + r) / 2
            this.cy = (t + b) / 2
            this.w = r - l
            this.h = b - t
        }
    }
    move(dx, dy) {
        this.cx = this.cx + dx
        this.cy = this.cy + dy
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
        return {
            "cls": this.cls,
            "cx": clamp(this.cx / w, 0.0, 1,0),
            "cy": clamp(this.cy / h, 0.0, 1,0),
            "w": clamp(this.w / w, 0.0, 1,0),
            "h": clamp(this.h / h, 0.0, 1,0),
        }
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
            clsChars : "가나다라마거너더러머고노도로모구누두루무버서어저보소오조부수우주허하호바사아자배울대광산경기충북전제인천세종강원남0123456789",
            isShowCls: true,

            annoData: [],

            canvas: null,
            width: 288, // Do not set directly, call updateView()
            height: 96, // Do not set directly, call updateView()

            image: null,
            scale: 1.0, // image scale
            viewScale: 3.0,
            viewRect: { l: 0, t: 0, w: 288, h: 96 }, // image view rect
            
            labels: [
                //new Label().setByValues(10, 10, 15, 100, 120, 80, 90, 33).move(100, 100)
            ],
            curLabel: null,
            curSide: null, // 0=up, 1=right, 2=bottom, 3=left

            isMouseProcessing: false,
            mx: 0,
            my: 0,

            isLabelCreating: false,
            creatingLabel: [],

            submitResult: "",
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
            for (let label of this.annoData.bbList) {
                let nl = new BbLabel(label.cls, label.cx * w, label.cy * h, label.w * w, label.h * h)
                this.labels.push(nl)
            }
            this.sortLabels()
        },

        onStart() {
            var vm = this
            this.$http.get('/ocrannos')
                .then(function(response) {
                    vm.annoData = response.data
                    vm.scale = 1.0
                    vm.curLabel = null
                    vm.curSide = null
                    vm.submitResult = ""
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
                "bbList": this.getLabelsJson()
            }

            let url = '/ocrannos'
            var vm = this
            this.$http.post(url, postData)
                .then(function (response) {
                    vm.showSubmitResult(response.status)
                })
                .catch(function (error) {
                    console.log(error.response)
                    vm.showSubmitResult(error.response.status)
                });
        },

        onSubmit() {
            this.submit()
            this.setFocusToCanvas()
        },

        showSubmitResult(code) {
            if (code === 200) {
                this.submitResult = "OK ( updated again!)"
            }
            else if (code === 201) {
                this.submitResult = "OK ( created )"
            }
            else if (code === 404) {
                this.submitResult = "Fail (Already submitted)"
            }
            else {
                this.submitResult = "Fail (code=" + code.toString() + ")"
            }
        },

        // canvas coordinate -> image coordinate
        toImageCoord(vx, vy) {
            let vr = this.viewRect
            let ix = vx / this.width * vr.w + vr.l
            let iy = vy / this.height * vr.h + vr.t
            return new Point(ix, iy)
        },
        // image coordinate -> canvas coordinate
        fromImageCoord(ix, iy) {
            let vr = this.viewRect
            let vx = (ix - vr.l) * this.width / vr.w
            let vy = (iy - vr.t) * this.height / vr.h
            return new Point(vx, vy)
        },

        getClassChar(cls) {
            return this.clsChars.charAt(cls)
        },

        // onMouseRClick(e) {
        //     //console.log("onMouseRClick", e)
        //     var pt = this.toImageCoord(e.offsetX, e.offsetY)
        // },

        onMouseLClick(e) {
            console.log("onMouseLClick", e)

            var pt = this.toImageCoord(e.offsetX, e.offsetY)

            if (this.isLabelCreating === true) {
                var cl = this.creatingLabel
                cl.push(pt)
                if (this.creatingLabel.length == 2) {
                    let l = Math.min(cl[0].x, cl[1].x)
                    let r = Math.max(cl[0].x, cl[1].x)
                    let t = Math.min(cl[0].y, cl[1].y)
                    let b = Math.max(cl[0].y, cl[1].y)
                    let bbLabel = new BbLabel(0, (l + r) / 2, (t + b) / 2, r - l, b - t)
                    this.labels.push(bbLabel)
                    this.isLabelCreating = false
                    this.creatingLabel = []
                    this.curLabel = bbLabel
                    this.curSide = 0
                    this.sortLabels()
                }
                this.updateView()
                return
            }

            if (e.altKey === true) {
                if (this.curLabel != null) {
                    if (this.curSide != null) {
                        if (this.curSide == 0) { this.curLabel.tset(pt.y) }
                        if (this.curSide == 1) { this.curLabel.rset(pt.x) }
                        if (this.curSide == 2) { this.curLabel.bset(pt.y) }
                        if (this.curSide == 3) { this.curLabel.lset(pt.x) }
                    }
                    else {
                        let dd = [
                            Math.abs(pt.y - this.curLabel.t()),
                            Math.abs(pt.x - this.curLabel.r()),
                            Math.abs(pt.y - this.curLabel.b()),
                            Math.abs(pt.x - this.curLabel.l())
                        ]
                        let i = dd.indexOf(Math.min(...dd))
                        if (i == 0) { this.curLabel.tset(pt.y, 20) }
                        if (i == 1) { this.curLabel.rset(pt.x, 20) }
                        if (i == 2) { this.curLabel.bset(pt.y, 20) }
                        if (i == 3) { this.curLabel.lset(pt.x, 20) }
                    }
                    this.updateView()
                }
                return
            }

            if (this.labels.length > 0) {
                // select including label
                this.curLabel = null
                this.curSide = null
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

        onKeyDown(e) {
            //console.log(e)

            if (this.isMouseProcessng === true) {
                if (e.key == 'Escape') {
                    if (this.isMouseProcessing) {
                        this.isMouseProcessing = false
                    }
                }
                return
            }

            if (this.isLabelCreating === true) {
                if (e.key == 'Escape') {
                    this.isLabelCreating = false
                    this.creatingLabel = []
                    this.updateView()
                }
                return
            }

            if (e.key == '`' || e.key == '~') {
                // navigate among sides
                let idx = null
                let labelIdx = this.labels.indexOf(this.curLabel)
                if (labelIdx < 0) {
                    idx = 0
                } else {
                    if (this.curSide == null) {
                        idx = labelIdx * 4
                    } else {
                        let diff = (e.key == '`' ? 1 : -1)
                        idx = labelIdx * 4 + this.curSide + diff
                    }
                }
                idx = (idx + this.labels.length * 4) % (this.labels.length * 4)

                this.curLabel = this.labels[Math.floor(idx / 4)]
                this.curSide = idx % 4
                this.updateView()
                return
            }

            if (e.key == '1' || e.key == '!') {
                let labelIdx = this.labels.indexOf(this.curLabel)
                if (labelIdx < 0) {
                    labelIdx = 0
                } else {
                    let diff = (e.key == '1' ? 1 : -1)
                    labelIdx = labelIdx + diff

                }
                labelIdx = (labelIdx + this.labels.length) % this.labels.length

                this.curLabel = this.labels[labelIdx]
                this.curSide = 0
                this.updateView()
                return
            }

            if (e.key == 'x') {
                this.isShowCls = !this.isShowCls
                this.updateView()
                return
            }

            if (e.key == 'c') {
                // label creating
                this.isLabelCreating = true
                this.creatingLabel = []
                this.curLabel = null
                this.curSide = null
                return
            }

            if (this.curLabel != null) {
                if (e.key == 'Escape') {
                    // deselect
                    this.curLabel = null
                    this.curSide = null
                }
                if (e.key == 'd') {
                    // delete selected label
                    var idx = this.labels.indexOf(this.curLabel)
                    if (idx >= 0) {
                        this.labels.splice(idx, 1)
                        this.curSide = null
                    }
                }
                if (e.key == "ArrowLeft") {
                    if (this.curSide == 1) this.curLabel.rset(this.curLabel.r() - 1)
                    if (this.curSide == 3) this.curLabel.lset(this.curLabel.l() - 1)
                }
                if (e.key == "ArrowRight") {
                    if (this.curSide == 1) this.curLabel.rset(this.curLabel.r() + 1)
                    if (this.curSide == 3) this.curLabel.lset(this.curLabel.l() + 1)
                }
                if (e.key == "ArrowUp") {
                    if (this.curSide == 0) this.curLabel.tset(this.curLabel.t() - 1)
                    if (this.curSide == 2) this.curLabel.bset(this.curLabel.b() - 1)
                }
                if (e.key == "ArrowDown") {
                    if (this.curSide == 0) this.curLabel.tset(this.curLabel.t() + 1)
                    if (this.curSide == 2) this.curLabel.bset(this.curLabel.b() + 1)
                }
                this.sortLabels()
                this.updateView()
                return
            }
        },

        sortLabels() {
            this.labels.sort(function(a, b) {
                return a.cx < b.cx ? -1 : (a.cx > b.cx ? 1 : 0)
            })
        },

        drawLabel(pts, clsChar, labelCurSide, lineColor="red", ptColor="blue") {
            if (pts.length == 4) {
                pts.push(pts[0].copy())
            }

            var vm = this
            pts = pts.map(function(pt) { 
                return vm.fromImageCoord(pt.x, pt.y)
            })

            var ctx = this.canvas

            // draw label
            ctx.lineWidth = "1"
            for (var i = 0; i < pts.length - 1; i++) {
                ctx.strokeStyle = labelCurSide === i ? "blue" : lineColor

                ctx.beginPath()
                ctx.moveTo(pts[i].x, pts[i].y)
                ctx.lineTo(pts[i+1].x, pts[i+1].y)
                ctx.stroke()
            }

            // draw first point
            ctx.fillStyle = ptColor
            ctx.fillRect(pts[0].x - 2, pts[0].y - 2, 4, 4)

            // dtaw cls text
            if (this.isShowCls) {
                ctx.textAlign = "left"
                ctx.textBaseline = "top"
                ctx.font = "28px malgun gothic"
                ctx.fillStyle = "rgba(255, 0, 255, 180)"
                ctx.fillText(clsChar, pts[0].x + 2, pts[0].y + 2)
            }
        },

        drawCreatingLabel(pts, ptColor="blue") {
            // draw first point
            let pt0 = this.fromImageCoord(pts[0].x, pts[0].y)

            var ctx = this.canvas
            ctx.fillStyle = ptColor
            ctx.fillRect(pt0.x - 2, pt0.y - 2, 4, 4)
        },

        updateView() {
            if (this.canvas === null || this.image === null) {
                return
            }

            let vr = this.viewRect
            this.width = vr.w * this.scale * this.viewScale
            this.height = vr.h * this.scale * this.viewScale

            this.$nextTick(() => {
                this.canvas.drawImage(this.image,
                    vr.l, vr.t, vr.w, vr.h,
                    0, 0, this.width, this.height)
                
                for (let label of this.labels) {
                    //console.log("label:", label)
                    let labelLineColor = (label === this.curLabel ? "yellow" : "red")
                    let labelCurSide = (label === this.curLabel ? this.curSide : null)
                    this.drawLabel(label.pts(), this.getClassChar(label.cls), labelCurSide, labelLineColor)
                }

                if (this.isLabelCreating) {
                    this.drawCreatingLabel(this.creatingLabel)
                }
            })
        },

        setViewRect(l, t, w, h) {
            this.viewRect.l = l
            this.viewRect.t = t
            this.viewRect.w = w
            this.viewRect.h = h
        },

        setCls(cls) {
            if (this.curLabel != null) {
                this.curLabel.cls = cls
                this.updateView()
            }
            this.setFocusToCanvas()
        },

        setFocusToCanvas() {
            document.getElementById("imgCanvas").focus()
        },

        onCls00() { this.setCls(0) },
        onCls01() { this.setCls(1) },
        onCls02() { this.setCls(2) },
        onCls03() { this.setCls(3) },
        onCls04() { this.setCls(4) },
        onCls05() { this.setCls(5) },
        onCls06() { this.setCls(6) },
        onCls07() { this.setCls(7) },
        onCls08() { this.setCls(8) },
        onCls09() { this.setCls(9) },
        onCls10() { this.setCls(10) },
        onCls11() { this.setCls(11) },
        onCls12() { this.setCls(12) },
        onCls13() { this.setCls(13) },
        onCls14() { this.setCls(14) },
        onCls15() { this.setCls(15) },
        onCls16() { this.setCls(16) },
        onCls17() { this.setCls(17) },
        onCls18() { this.setCls(18) },
        onCls19() { this.setCls(19) },
        onCls20() { this.setCls(20) },
        onCls21() { this.setCls(21) },
        onCls22() { this.setCls(22) },
        onCls23() { this.setCls(23) },
        onCls24() { this.setCls(24) },
        onCls25() { this.setCls(25) },
        onCls26() { this.setCls(26) },
        onCls27() { this.setCls(27) },
        onCls28() { this.setCls(28) },
        onCls29() { this.setCls(29) },
        onCls30() { this.setCls(30) },
        onCls31() { this.setCls(31) },
        onCls32() { this.setCls(32) },
        onCls33() { this.setCls(33) },
        onCls34() { this.setCls(34) },
        onCls35() { this.setCls(35) },
        onCls36() { this.setCls(36) },
        onCls37() { this.setCls(37) },
        onCls38() { this.setCls(38) },
        onCls39() { this.setCls(39) },
        onCls40() { this.setCls(40) },
        onCls41() { this.setCls(41) },
        onCls42() { this.setCls(42) },
        onCls43() { this.setCls(43) },
        onCls44() { this.setCls(44) },
        onCls45() { this.setCls(45) },
        onCls46() { this.setCls(46) },
        onCls47() { this.setCls(47) },
        onCls48() { this.setCls(48) },
        onCls49() { this.setCls(49) },
        onCls50() { this.setCls(50) },
        onCls51() { this.setCls(51) },
        onCls52() { this.setCls(52) },
        onCls53() { this.setCls(53) },
        onCls54() { this.setCls(54) },
        onCls55() { this.setCls(55) },
        onCls56() { this.setCls(56) },
        onCls57() { this.setCls(57) },
        onCls58() { this.setCls(58) },
        onCls59() { this.setCls(59) },
        onCls60() { this.setCls(60) },
        onCls61() { this.setCls(61) },
        onCls62() { this.setCls(62) },
        onCls63() { this.setCls(63) },
        onCls64() { this.setCls(64) },
        onCls65() { this.setCls(65) },
        onCls66() { this.setCls(66) },
    },
}
</script>

<style scoped>

#imgCanvas {
    border: 1px solid grey;
}

</style>

