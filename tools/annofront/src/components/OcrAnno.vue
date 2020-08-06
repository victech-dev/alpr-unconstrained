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
                        file : {{ annoData === null ? "none" : annoData.file }}
                    </h4>
                </v-row>
            </v-col>

            <v-dialog
                v-model="showHelp"
                max-width="800"
                >
                <v-card>
                    <v-card-title class="headline">
                        Ocr 데이터 편집 가이드
                    </v-card-title>

                    <v-card-text>
                        <v-divider></v-divider>
                        <br>
                        <p>
                            'START' 버튼으로 이미지 한 장을 받아옵니다.<br>
                            번호판 각각의 글자에 대해서 bouding-box(bb)를 라벨링 해야 합니다.<br>
                            또한 해당 bb가 어느 글자인지 지정해 주어야 합니다. (bb선택 후 글자 버튼 클릭)<br>
                            이후 'SUBMIT' 으로 제출합니다. <b>(여러번 submit 가능!!)</b>
                        </p>
                        <v-divider></v-divider>
                        <br>
                        <h3>마우스 클릭</h3>
                        <p>&nbsp;&nbsp;그 곳의 bb를 선택</p>
                        <h3>'1'</h3>
                        <p>&nbsp;&nbsp;왼쪽->오른쪽으로 선택된 bb를 옮깁니다. shift는 반대 방향</p>
                        <h3>'`'</h3>
                        <p>&nbsp;&nbsp;왼쪽->오른쪽, t->r->b->l 순서로 side를 선택합니다. shift는 반대 방향</p>
                        <h3>'방향키'</h3>
                        <p>&nbsp;&nbsp;현재 선택된 side가 있으면 그 위치를 옮깁니다</p>
                        <h3>'alt' + 마우스클릭</h3>
                        <p>&nbsp;&nbsp;선택된 side가 있으면 그 위치를 옮기고<br>
                           &nbsp;&nbsp;bb만 선택되어 있으면 가장 가까운 side를 옮겨줍니다.</p>
                        <h3>'c'</h3>
                        <p>&nbsp;&nbsp;bounding-box 추가 시작. 이후 lt/rb 두 번 클릭하면 bb가 추가</p>
                        <h3>'x'</h3>
                        <p>&nbsp;&nbsp;글자 보여줌/안보여줌 토글</p>

                        <v-divider></v-divider>
                        
                    </v-card-text>

                    <v-card-actions>
                        <v-spacer></v-spacer>
                        <v-btn color="blue darken-1" text @click="showHelp=false;setFocusToCanvas()">Close</v-btn>
                    </v-card-actions>
                </v-card>
            </v-dialog>
        </v-row>
        <v-row align="start" justify="center" >
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls00">가</v-btn>
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls01">나</v-btn>
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls02">다</v-btn>
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls03">라</v-btn>
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls04">마</v-btn>
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls35">바</v-btn>
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls36">사</v-btn>
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls37">아</v-btn>
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls38">자</v-btn>
        </v-row>
        <v-row align="start" justify="center" >
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls05">거</v-btn>
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls06">너</v-btn>
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls07">더</v-btn>
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls08">러</v-btn>
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls09">머</v-btn>
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls20">버</v-btn>
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls21">서</v-btn>
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls22">어</v-btn>
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls23">저</v-btn>
        </v-row>
        <v-row align="start" justify="center" >
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls10">고</v-btn>
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls11">노</v-btn>
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls12">도</v-btn>
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls13">로</v-btn>
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls14">모</v-btn>
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls24">보</v-btn>
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls25">소</v-btn>
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls26">오</v-btn>
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls27">조</v-btn>
        </v-row>
        <v-row align="start" justify="center" >
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls15">구</v-btn>
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls16">누</v-btn>
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls17">두</v-btn>
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls18">루</v-btn>
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls19">무</v-btn>
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls28">부</v-btn>
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls29">수</v-btn>
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls30">우</v-btn>
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls31">주</v-btn>
        </v-row>
        <v-row align="start" justify="center" >
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls32">허</v-btn>
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls33">하</v-btn>
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls34">호</v-btn>
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls39">배</v-btn>
        </v-row>
        <v-row align="start" justify="center" >
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls40">울</v-btn>
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls41">대</v-btn>
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls42">광</v-btn>
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls43">산</v-btn>
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls44">경</v-btn>
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls45">기</v-btn>
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls46">충</v-btn>
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls47">북</v-btn>
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls48">전</v-btn>
        </v-row>
        <v-row align="start" justify="center" >
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls49">제</v-btn>
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls50">인</v-btn>
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls51">천</v-btn>
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls52">세</v-btn>
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls53">종</v-btn>
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls54">강</v-btn>
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls55">원</v-btn>
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls56">남</v-btn>
        </v-row>
        <v-row align="start" justify="center" >
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls57">0</v-btn>
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls58">1</v-btn>
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls59">2</v-btn>
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls60">3</v-btn>
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls61">4</v-btn>
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls62">5</v-btn>
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls63">6</v-btn>
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls64">7</v-btn>
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls65">8</v-btn>
            <v-btn class="ma-1 pa-0" tile outlined color="blue-grey" small v-on:click="onCls66">9</v-btn>
        </v-row>
        <v-row align="start" justify="center">
            <canvas
                id="imgCanvas" 
                tabindex="0" 
                v-bind:width="width" 
                v-bind:height="height" 
                @keydown="onKeyDown" 
                @click="onMouseLClick"/>
        </v-row>

        <v-snackbar
            v-model="submitResultShow"
            :color="submitResultColor"
            timeout=6000
            bottom=true
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
            width: 288 * 3, // Do not set directly, call updateView()
            height: 96 * 3, // Do not set directly, call updateView()

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

