(function(t){function s(s){for(var n,o,a=s[0],u=s[1],c=s[2],r=0,b=[];r<a.length;r++)o=a[r],Object.prototype.hasOwnProperty.call(i,o)&&i[o]&&b.push(i[o][0]),i[o]=0;for(n in u)Object.prototype.hasOwnProperty.call(u,n)&&(t[n]=u[n]);h&&h(s);while(b.length)b.shift()();return l.push.apply(l,c||[]),e()}function e(){for(var t,s=0;s<l.length;s++){for(var e=l[s],n=!0,a=1;a<e.length;a++){var u=e[a];0!==i[u]&&(n=!1)}n&&(l.splice(s--,1),t=o(o.s=e[0]))}return t}var n={},i={app:0},l=[];function o(s){if(n[s])return n[s].exports;var e=n[s]={i:s,l:!1,exports:{}};return t[s].call(e.exports,e,e.exports,o),e.l=!0,e.exports}o.m=t,o.c=n,o.d=function(t,s,e){o.o(t,s)||Object.defineProperty(t,s,{enumerable:!0,get:e})},o.r=function(t){"undefined"!==typeof Symbol&&Symbol.toStringTag&&Object.defineProperty(t,Symbol.toStringTag,{value:"Module"}),Object.defineProperty(t,"__esModule",{value:!0})},o.t=function(t,s){if(1&s&&(t=o(t)),8&s)return t;if(4&s&&"object"===typeof t&&t&&t.__esModule)return t;var e=Object.create(null);if(o.r(e),Object.defineProperty(e,"default",{enumerable:!0,value:t}),2&s&&"string"!=typeof t)for(var n in t)o.d(e,n,function(s){return t[s]}.bind(null,n));return e},o.n=function(t){var s=t&&t.__esModule?function(){return t["default"]}:function(){return t};return o.d(s,"a",s),s},o.o=function(t,s){return Object.prototype.hasOwnProperty.call(t,s)},o.p="/";var a=window["webpackJsonp"]=window["webpackJsonp"]||[],u=a.push.bind(a);a.push=s,a=a.slice();for(var c=0;c<a.length;c++)s(a[c]);var h=u;l.push([0,"chunk-vendors"]),e()})({0:function(t,s,e){t.exports=e("56d7")},"034f":function(t,s,e){"use strict";var n=e("85ec"),i=e.n(n);i.a},1918:function(t,s,e){"use strict";var n=e("cc9a"),i=e.n(n);i.a},"233b":function(t,s,e){t.exports=e.p+"img/sample.5ca5980d.png"},"43d0":function(t,s,e){},"56d7":function(t,s,e){"use strict";e.r(s);e("e260"),e("e6cf"),e("cca6"),e("a79d");var n=e("2b0e"),i=function(){var t=this,s=t.$createElement,e=t._self._c||s;return e("div",{attrs:{id:"app"}},[e("p",[e("router-link",{attrs:{to:"/",tag:"button"}},[t._v("Wpod Editor")]),t._v("-- "),e("router-link",{attrs:{to:"/ocranno",tag:"button"}},[t._v("Ocr Editor")]),t._v("-- "),e("router-link",{attrs:{to:"/help",tag:"button"}},[t._v("Help")])],1),e("router-view")],1)},l=[],o={name:"App",components:{}},a=o,u=(e("034f"),e("2877")),c=Object(u["a"])(a,i,l,!1,null,null,null),h=c.exports,r=e("8c4f"),b=function(){var t=this,s=t.$createElement,e=t._self._c||s;return e("div",{staticClass:"wpodanno"},[e("div",[e("button",{staticClass:"pure-button",on:{click:t.onStart}},[t._v("Start")]),t._v("-- "),e("button",{staticClass:"pure-button",on:{click:t.onSubmit}},[t._v("Submit")]),t._v("-- "+t._s(t.submitResult)+" ")]),e("h5",[t._v(" file : "+t._s(null===t.annoData?"none":t.annoData.file)+", wh=("+t._s(null===t.image?0:t.image.width)+", "+t._s(null===t.image?0:t.image.height)+") ")]),e("canvas",{attrs:{id:"imgCanvas",tabindex:"0",width:t.width,height:t.height},on:{keydown:t.onKeyDown,mousedown:t.onMouseDown,mouseup:t.onMouseUp,click:t.onMouseClick,dblclick:t.onMouseDbClick}})])},f=[],v=(e("c975"),e("d81d"),e("13d5"),e("a434"),e("d3b7"),e("25f0"),e("b85c")),C=(e("96cf"),e("1da1")),p=e("d4ec"),d=e("bee2");function g(t,s,e){return t<=s?s:t>=e?e:t}function y(t){return console.log(t),new Promise((function(s){var e=new Image;e.onload=function(){return s(e)},e.src=t}))}var w=function(){function t(){var s=arguments.length>0&&void 0!==arguments[0]?arguments[0]:0,e=arguments.length>1&&void 0!==arguments[1]?arguments[1]:0;Object(p["a"])(this,t),this.x=s,this.y=e}return Object(d["a"])(t,[{key:"set",value:function(t,s){return this.x=t,this.y=s,this}},{key:"fromXY",value:function(s,e){return new t(s,e)}},{key:"move",value:function(t,s){return this.x+=t,this.y+=s,this}},{key:"copy",value:function(){return new t(this.x,this.y)}},{key:"sqrdist",value:function(t){return Math.pow(Math.abs(t.x-this.x),2)+Math.pow(Math.abs(t.y-this.y),2)}},{key:"dist",value:function(t){return Math.sqrt(this.sqrdist(t))}},{key:"getJson",value:function(t,s){return{x:g(this.x/t,0,1,0),y:g(this.y/s,0,1)}}}]),t}(),m=function(){function t(){var s=arguments.length>0&&void 0!==arguments[0]?arguments[0]:new w,e=arguments.length>1&&void 0!==arguments[1]?arguments[1]:new w,n=arguments.length>2&&void 0!==arguments[2]?arguments[2]:new w,i=arguments.length>3&&void 0!==arguments[3]?arguments[3]:new w;Object(p["a"])(this,t),this.pt0=s,this.pt1=e,this.pt2=n,this.pt3=i}return Object(d["a"])(t,[{key:"setByPts",value:function(t,s,e,n){return this.pt0=t,this.pt1=s,this.pt2=e,this.pt3=n,this}},{key:"setByValues",value:function(t,s,e,n,i,l,o,a){return this.pt0.set(t,s),this.pt1.set(e,n),this.pt2.set(i,l),this.pt3.set(o,a),this}},{key:"copy",value:function(){return new t(this.pt0.copy(),this.pt1.copy(),this.pt2.copy(),this.pt3.copy())}},{key:"pts",value:function(){return[this.pt0,this.pt1,this.pt2,this.pt3]}},{key:"pt",value:function(t){return 0==t?this.pt0:1==t?this.pt1:2==t?this.pt2:3==t?this.pt3:null}},{key:"getBb",value:function(){var t=Math.min.apply(Math,this.pts().map((function(t){return t.x}))),s=Math.max.apply(Math,this.pts().map((function(t){return t.x}))),e=Math.min.apply(Math,this.pts().map((function(t){return t.y}))),n=Math.max.apply(Math,this.pts().map((function(t){return t.y})));return[t,s,e,n]}},{key:"move",value:function(t,s){return this.pts().map((function(e){e.x+=t,e.y+=s})),this}},{key:"fineNearest",value:function(t){return this.pts().map((function(t,s){return[t,s]})).reduce((function(s,e){return t.sqrdist(s[0])<t.sqrdist(e[0])?s:e}))}},{key:"include",value:function(t){for(var s=this.pts(),e=!1,n=0,i=s.length-1;n<s.length;i=n++){var l=s[n].x,o=s[n].y,a=s[i].x,u=s[i].y,c=o>t.y!=u>t.y&&t.x<(a-l)*(t.y-o)/(u-o)+l;c&&(e=!e)}return e}},{key:"getJson",value:function(t,s){return[this.pt0.getJson(t,s),this.pt1.getJson(t,s),this.pt2.getJson(t,s),this.pt3.getJson(t,s)]}}]),t}(),k={created:function(){},mounted:function(){console.log("mounted");var t=document.getElementById("imgCanvas");this.canvas=t.getContext("2d"),this.canvas.imageSmoothingEnabled=!1},data:function(){return{annoData:[],canvas:null,width:500,height:500,image:null,scale:1,viewRect:{l:0,t:0,w:500,h:500},labels:[],curLabel:null,isMouseProcessing:!1,mx:0,my:0,isLabelCreating:!1,creatingLabel:[],submitResult:""}},methods:{handleNewData:function(){var t=this;return Object(C["a"])(regeneratorRuntime.mark((function s(){var e;return regeneratorRuntime.wrap((function(s){while(1)switch(s.prev=s.next){case 0:return t.labels=[],s.next=3,y(t.annoData.file);case 3:e=s.sent,t.image=e,t.createLabels(),t.setViewRect(0,0,e.width,e.height),t.updateView();case 8:case"end":return s.stop()}}),s)})))()},createLabels:function(){var t,s=this.image.width,e=this.image.height,n=Object(v["a"])(this.annoData.labels);try{for(n.s();!(t=n.n()).done;){var i=t.value;if(4==i.length){for(var l=[],o=0;o<i.length;++o)l.push(new w(i[o].x*s,i[o].y*e));this.labels.push(new m(l[0],l[1],l[2],l[3]))}}}catch(a){n.e(a)}finally{n.f()}},onStart:function(){var t=this;this.$http.get("/wpodannos").then((function(s){t.annoData=s.data,t.scale=1,t.curLabel=null,t.handleNewData()}))},getLabelsJson:function(){var t,s=[],e=Object(v["a"])(this.labels);try{for(e.s();!(t=e.n()).done;){var n=t.value;s.push(n.getJson(this.image.width,this.image.height))}}catch(i){e.e(i)}finally{e.f()}return s},onSubmit:function(){if(!0!==this.isLabelCreating&&!0!==this.isMouseProcessing)if(0!==this.labels.length){var t={file:this.annoData.file,labels:this.getLabelsJson()},s=this;this.$http.post("/wpodannos",t).then((function(t){console.log(t),s.showSubmitResult(t.status)})).catch((function(t){console.log(t.response),s.showSubmitResult(t.response.status)}))}else alert("No label data!");else alert("Cannot now!")},showSubmitResult:function(t){var s=this;this.submitResult=200===t?"Success":404===t?"Fail (Already submitted)":"Fail (code="+t.toString()+")",setTimeout((function(){s.submitResult=""}),5e3)},toImageCoord:function(t,s){var e=this.viewRect,n=t/this.width*e.w+e.l,i=s/this.height*e.h+e.t;return new w(n,i)},fromImageCoord:function(t,s){var e=this.viewRect,n=(t-e.l)*this.width/e.w,i=(s-e.t)*this.height/e.h;return new w(n,i)},onMouseDown:function(t){this.mx=t.offsetX,this.my=t.offsetY,this.isMouseProcessing=!0},onMouseUp:function(t){if(!0===this.isMouseProcessing&&(this.isMouseProcessing=!1,Math.abs(this.mx-t.offsetX)>=10&&Math.abs(this.my-t.offsetY)>=10)){var s=Math.min(this.mx,t.offsetX),e=Math.max(this.mx,t.offsetX),n=Math.min(this.my,t.offsetY),i=Math.max(this.my,t.offsetY),l=this.toImageCoord(s,n),o=this.toImageCoord(e,i);this.setViewRect(l.x,l.y,o.x-l.x,o.y-l.y),this.scale=this.image.width/(o.x-l.x),this.updateView()}},onMouseClick:function(t){var s=this.toImageCoord(t.offsetX,t.offsetY);if(!0===this.isLabelCreating){var e=this.creatingLabel;return e.push(s),4==this.creatingLabel.length&&(this.labels.push(new m(e[0],e[1],e[2],e[3])),this.isLabelCreating=!1,this.creatingLabel=[]),void this.updateView()}if(null==this.curLabel);else{var n=this.curLabel.fineNearest(s);n[0].dist(s)<100&&(n[0].set(s.x,s.y),this.updateView())}},onMouseDbClick:function(t){var s=this.toImageCoord(t.offsetX,t.offsetY);if(!0!==this.isLabelCreating){if(null!=this.curLabel)return this.curLabel=null,void this.updateView();if(this.labels.length>0){this.curLabel=null;for(var e=0;e<this.labels.length;e++)if(1==this.labels[e].include(s)){this.curLabel=this.labels[e];break}this.updateView()}else;}},onKeyDown:function(t){if(!0===this.isMouseProcessng&&"Escape"==t.key&&this.isMouseProcessing&&(this.isMouseProcessing=!1),!0===this.isLabelCreating&&"Escape"==t.key&&(this.isLabelCreating=!1,this.creatingLabel=[],this.updateView()),null!=this.curLabel&&"Escape"==t.key&&(this.curLabel=null,this.updateView()),"s"==t.key&&1==t.ctrlKey&&(this.scale-=t.altKey?.1:.5,this.updateView()),"`"==t.key&&(this.scale=1,this.setViewRect(0,0,this.image.width,this.image.height),this.updateView()),"c"==t.key&&(this.isLabelCreating=!0,this.creatingLabel=[]),"d"==t.key&&null!=this.curLabel){var s=this.labels.indexOf(this.curLabel);s>=0&&(this.labels.splice(s,1),this.updateView())}},drawLabel:function(t){var s=arguments.length>1&&void 0!==arguments[1]?arguments[1]:"red",e=arguments.length>2&&void 0!==arguments[2]?arguments[2]:"blue";4==t.length&&t.push(t[0].copy());var n=this;t=t.map((function(t){return n.fromImageCoord(t.x,t.y)}));var i=this.canvas;i.lineWidth="1",i.strokeStyle=s,i.beginPath(),i.moveTo(t[0].x,t[0].y);for(var l=1;l<t.length;l++)i.lineTo(t[l].x,t[l].y);i.stroke(),i.fillStyle=e,i.fillRect(t[0].x-2,t[0].y-2,4,4)},updateView:function(){var t=this;if(null!==this.canvas&&null!==this.image){var s=this.viewRect;this.width=s.w*this.scale,this.height=s.h*this.scale,this.$nextTick((function(){t.canvas.drawImage(t.image,s.l,s.t,s.w,s.h,0,0,t.width,t.height);var e,n=Object(v["a"])(t.labels);try{for(n.s();!(e=n.n()).done;){var i=e.value,l=i===t.curLabel?"yellow":"red";t.drawLabel(i.pts(),l)}}catch(o){n.e(o)}finally{n.f()}t.isLabelCreating&&t.drawLabel(t.creatingLabel)}))}},setViewRect:function(t,s,e,n){this.viewRect.l=t,this.viewRect.t=s,this.viewRect.w=e,this.viewRect.h=n}}},_=k,L=(e("59a5"),Object(u["a"])(_,b,f,!1,null,"5414cc49",null)),x=L.exports,S=function(){var t=this,s=t.$createElement,e=t._self._c||s;return e("div",{staticClass:"ocranno"},[e("div",[e("button",{staticClass:"pure-button",on:{click:t.onStart}},[t._v("Start")]),t._v("-- "),e("button",{staticClass:"pure-button",on:{click:t.onSubmit}},[t._v("Submit")])]),e("div",[t._v(" -- "+t._s(t.submitResult)+" -- ")]),e("h5",[t._v(" file : "+t._s(null===t.annoData?"none":t.annoData.file)+", wh=("+t._s(null===t.image?0:t.image.width)+", "+t._s(null===t.image?0:t.image.height)+") ")]),e("div",[e("button",{staticClass:"pure-button",on:{click:t.onCls00}},[t._v("가")]),e("button",{staticClass:"pure-button",on:{click:t.onCls01}},[t._v("나")]),e("button",{staticClass:"pure-button",on:{click:t.onCls02}},[t._v("다")]),e("button",{staticClass:"pure-button",on:{click:t.onCls03}},[t._v("라")]),e("button",{staticClass:"pure-button",on:{click:t.onCls04}},[t._v("마")]),e("button",{staticClass:"pure-button",on:{click:t.onCls35}},[t._v("바")]),e("button",{staticClass:"pure-button",on:{click:t.onCls36}},[t._v("사")]),e("button",{staticClass:"pure-button",on:{click:t.onCls37}},[t._v("아")]),e("button",{staticClass:"pure-button",on:{click:t.onCls38}},[t._v("자")])]),e("div",[e("button",{staticClass:"pure-button",on:{click:t.onCls05}},[t._v("거")]),e("button",{staticClass:"pure-button",on:{click:t.onCls06}},[t._v("너")]),e("button",{staticClass:"pure-button",on:{click:t.onCls07}},[t._v("더")]),e("button",{staticClass:"pure-button",on:{click:t.onCls08}},[t._v("러")]),e("button",{staticClass:"pure-button",on:{click:t.onCls09}},[t._v("머")]),e("button",{staticClass:"pure-button",on:{click:t.onCls20}},[t._v("버")]),e("button",{staticClass:"pure-button",on:{click:t.onCls21}},[t._v("서")]),e("button",{staticClass:"pure-button",on:{click:t.onCls22}},[t._v("어")]),e("button",{staticClass:"pure-button",on:{click:t.onCls23}},[t._v("저")])]),e("div",[e("button",{staticClass:"pure-button",on:{click:t.onCls10}},[t._v("고")]),e("button",{staticClass:"pure-button",on:{click:t.onCls11}},[t._v("노")]),e("button",{staticClass:"pure-button",on:{click:t.onCls12}},[t._v("도")]),e("button",{staticClass:"pure-button",on:{click:t.onCls13}},[t._v("로")]),e("button",{staticClass:"pure-button",on:{click:t.onCls14}},[t._v("모")]),e("button",{staticClass:"pure-button",on:{click:t.onCls24}},[t._v("보")]),e("button",{staticClass:"pure-button",on:{click:t.onCls25}},[t._v("소")]),e("button",{staticClass:"pure-button",on:{click:t.onCls26}},[t._v("오")]),e("button",{staticClass:"pure-button",on:{click:t.onCls27}},[t._v("조")])]),e("div",[e("button",{staticClass:"pure-button",on:{click:t.onCls15}},[t._v("구")]),e("button",{staticClass:"pure-button",on:{click:t.onCls16}},[t._v("누")]),e("button",{staticClass:"pure-button",on:{click:t.onCls17}},[t._v("두")]),e("button",{staticClass:"pure-button",on:{click:t.onCls18}},[t._v("루")]),e("button",{staticClass:"pure-button",on:{click:t.onCls19}},[t._v("무")]),e("button",{staticClass:"pure-button",on:{click:t.onCls28}},[t._v("부")]),e("button",{staticClass:"pure-button",on:{click:t.onCls29}},[t._v("수")]),e("button",{staticClass:"pure-button",on:{click:t.onCls30}},[t._v("우")]),e("button",{staticClass:"pure-button",on:{click:t.onCls31}},[t._v("주")])]),e("div",[e("button",{staticClass:"pure-button",on:{click:t.onCls32}},[t._v("허")]),e("button",{staticClass:"pure-button",on:{click:t.onCls33}},[t._v("하")]),e("button",{staticClass:"pure-button",on:{click:t.onCls34}},[t._v("호")]),e("button",{staticClass:"pure-button",on:{click:t.onCls39}},[t._v("배")])]),e("div",[e("button",{staticClass:"pure-button",on:{click:t.onCls40}},[t._v("울")]),e("button",{staticClass:"pure-button",on:{click:t.onCls41}},[t._v("대")]),e("button",{staticClass:"pure-button",on:{click:t.onCls42}},[t._v("광")]),e("button",{staticClass:"pure-button",on:{click:t.onCls43}},[t._v("산")]),e("button",{staticClass:"pure-button",on:{click:t.onCls44}},[t._v("경")]),e("button",{staticClass:"pure-button",on:{click:t.onCls45}},[t._v("기")]),e("button",{staticClass:"pure-button",on:{click:t.onCls46}},[t._v("충")]),e("button",{staticClass:"pure-button",on:{click:t.onCls47}},[t._v("북")]),e("button",{staticClass:"pure-button",on:{click:t.onCls48}},[t._v("전")]),e("button",{staticClass:"pure-button",on:{click:t.onCls49}},[t._v("제")])]),e("div",[e("button",{staticClass:"pure-button",on:{click:t.onCls50}},[t._v("인")]),e("button",{staticClass:"pure-button",on:{click:t.onCls51}},[t._v("천")]),e("button",{staticClass:"pure-button",on:{click:t.onCls52}},[t._v("세")]),e("button",{staticClass:"pure-button",on:{click:t.onCls53}},[t._v("종")]),e("button",{staticClass:"pure-button",on:{click:t.onCls54}},[t._v("강")]),e("button",{staticClass:"pure-button",on:{click:t.onCls55}},[t._v("원")]),e("button",{staticClass:"pure-button",on:{click:t.onCls56}},[t._v("남")])]),e("div",[e("button",{staticClass:"pure-button",on:{click:t.onCls57}},[t._v("0")]),e("button",{staticClass:"pure-button",on:{click:t.onCls58}},[t._v("1")]),e("button",{staticClass:"pure-button",on:{click:t.onCls59}},[t._v("2")]),e("button",{staticClass:"pure-button",on:{click:t.onCls60}},[t._v("3")]),e("button",{staticClass:"pure-button",on:{click:t.onCls61}},[t._v("4")]),e("button",{staticClass:"pure-button",on:{click:t.onCls62}},[t._v("5")]),e("button",{staticClass:"pure-button",on:{click:t.onCls63}},[t._v("6")]),e("button",{staticClass:"pure-button",on:{click:t.onCls64}},[t._v("7")]),e("button",{staticClass:"pure-button",on:{click:t.onCls65}},[t._v("8")]),e("button",{staticClass:"pure-button",on:{click:t.onCls66}},[t._v("9")])]),e("h4",[t._v(" Char : "+t._s(null===t.curLabel?"-":t.getClassChar(t.curLabel.cls))+" ")]),e("canvas",{attrs:{id:"imgCanvas",tabindex:"0",width:t.width,height:t.height},on:{keydown:t.onKeyDown,click:t.onMouseLClick}})])},M=[];function R(t,s,e){return t<=s?s:t>=e?e:t}function O(t){return new Promise((function(s){var e=new Image;e.onload=function(){return s(e)},e.src=t}))}var V=function(){function t(){var s=arguments.length>0&&void 0!==arguments[0]?arguments[0]:0,e=arguments.length>1&&void 0!==arguments[1]?arguments[1]:0;Object(p["a"])(this,t),this.x=s,this.y=e}return Object(d["a"])(t,[{key:"set",value:function(t,s){return this.x=t,this.y=s,this}},{key:"fromXY",value:function(s,e){return new t(s,e)}},{key:"move",value:function(t,s){return this.x+=t,this.y+=s,this}},{key:"copy",value:function(){return new t(this.x,this.y)}},{key:"sqrdist",value:function(t){return Math.pow(Math.abs(t.x-this.x),2)+Math.pow(Math.abs(t.y-this.y),2)}},{key:"dist",value:function(t){return Math.sqrt(this.sqrdist(t))}},{key:"getJson",value:function(t,s){return{x:R(this.x/t,0,1,0),y:R(this.y/s,0,1)}}}]),t}(),j=function(){function t(){var s=arguments.length>0&&void 0!==arguments[0]?arguments[0]:-1,e=arguments.length>1&&void 0!==arguments[1]?arguments[1]:0,n=arguments.length>2&&void 0!==arguments[2]?arguments[2]:0,i=arguments.length>3&&void 0!==arguments[3]?arguments[3]:0,l=arguments.length>4&&void 0!==arguments[4]?arguments[4]:0;Object(p["a"])(this,t),this.cls=s,this.cx=e,this.cy=n,this.w=i,this.h=l}return Object(d["a"])(t,[{key:"copy",value:function(){return new t(this.cls,this.cx,this.cy,this.w,this.h)}},{key:"l",value:function(){return this.cx-this.w/2}},{key:"r",value:function(){return this.cx+this.w/2}},{key:"t",value:function(){return this.cy-this.h/2}},{key:"b",value:function(){return this.cy+this.h/2}},{key:"lset",value:function(t){var s=arguments.length>1&&void 0!==arguments[1]?arguments[1]:0;(s<=0||s>Math.abs(this.l()-t))&&this.setBb(t,this.t(),this.r(),this.b())}},{key:"rset",value:function(t){var s=arguments.length>1&&void 0!==arguments[1]?arguments[1]:0;(s<=0||s>Math.abs(this.r()-t))&&this.setBb(this.l(),this.t(),t,this.b())}},{key:"tset",value:function(t){var s=arguments.length>1&&void 0!==arguments[1]?arguments[1]:0;(s<=0||s>Math.abs(this.t()-t))&&this.setBb(this.l(),t,this.r(),this.b())}},{key:"bset",value:function(t){var s=arguments.length>1&&void 0!==arguments[1]?arguments[1]:0;(s<=0||s>Math.abs(this.b()-t))&&this.setBb(this.l(),this.t(),this.r(),t)}},{key:"pts",value:function(){return[new V(this.l(),this.t()),new V(this.r(),this.t()),new V(this.r(),this.b()),new V(this.l(),this.b())]}},{key:"pt",value:function(t){return 0==t?new V(this.l(),this.t()):1==t?new V(this.r(),this.t()):2==t?new V(this.r(),this.b()):3==t?new V(this.l(),this.b()):null}},{key:"getBb",value:function(){return[this.l(),this.r(),this.t(),this.b()]}},{key:"setBb",value:function(t,s,e,n){t<e&&s<n&&(this.cx=(t+e)/2,this.cy=(s+n)/2,this.w=e-t,this.h=n-s)}},{key:"move",value:function(t,s){return this.cx=this.cx+t,this.cy=this.cy+s,this}},{key:"fineNearest",value:function(t){return this.pts().map((function(t,s){return[t,s]})).reduce((function(s,e){return t.sqrdist(s[0])<t.sqrdist(e[0])?s:e}))}},{key:"include",value:function(t){for(var s=this.pts(),e=!1,n=0,i=s.length-1;n<s.length;i=n++){var l=s[n].x,o=s[n].y,a=s[i].x,u=s[i].y,c=o>t.y!=u>t.y&&t.x<(a-l)*(t.y-o)/(u-o)+l;c&&(e=!e)}return e}},{key:"getJson",value:function(t,s){return{cls:this.cls,cx:R(this.cx/t,0,1,0),cy:R(this.cy/s,0,1,0),w:R(this.w/t,0,1,0),h:R(this.h/s,0,1,0)}}}]),t}(),D={created:function(){},mounted:function(){var t=document.getElementById("imgCanvas");this.canvas=t.getContext("2d"),this.canvas.imageSmoothingEnabled=!1},data:function(){return{clsChars:"가나다라마거너더러머고노도로모구누두루무버서어저보소오조부수우주허하호바사아자배울대광산경기충북전제인천세종강원남0123456789",isShowCls:!0,annoData:[],canvas:null,width:288,height:96,image:null,scale:1,viewScale:3,viewRect:{l:0,t:0,w:288,h:96},labels:[],curLabel:null,curSide:null,isMouseProcessing:!1,mx:0,my:0,isLabelCreating:!1,creatingLabel:[],submitResult:""}},methods:{handleNewData:function(){var t=this;return Object(C["a"])(regeneratorRuntime.mark((function s(){var e;return regeneratorRuntime.wrap((function(s){while(1)switch(s.prev=s.next){case 0:return t.labels=[],s.next=3,O(t.annoData.file);case 3:e=s.sent,t.image=e,t.createLabels(),t.setViewRect(0,0,e.width,e.height),t.updateView(),t.setFocusToCanvas();case 9:case"end":return s.stop()}}),s)})))()},createLabels:function(){var t,s=this.image.width,e=this.image.height,n=Object(v["a"])(this.annoData.bbList);try{for(n.s();!(t=n.n()).done;){var i=t.value,l=new j(i.cls,i.cx*s,i.cy*e,i.w*s,i.h*e);this.labels.push(l)}}catch(o){n.e(o)}finally{n.f()}this.sortLabels()},onStart:function(){var t=this;this.$http.get("/ocrannos").then((function(s){t.annoData=s.data,t.scale=1,t.curLabel=null,t.curSide=null,t.submitResult="",t.handleNewData()}))},getLabelsJson:function(){var t,s=[],e=Object(v["a"])(this.labels);try{for(e.s();!(t=e.n()).done;){var n=t.value;s.push(n.getJson(this.image.width,this.image.height))}}catch(i){e.e(i)}finally{e.f()}return s},submit:function(){if(!0!==this.isLabelCreating&&!0!==this.isMouseProcessing)if(0!==this.labels.length){var t={file:this.annoData.file,bbList:this.getLabelsJson()},s="/ocrannos",e=this;this.$http.post(s,t).then((function(t){e.showSubmitResult(t.status)})).catch((function(t){console.log(t.response),e.showSubmitResult(t.response.status)}))}else alert("No label data!");else alert("Cannot now!")},onSubmit:function(){this.submit(),this.setFocusToCanvas()},showSubmitResult:function(t){this.submitResult=200===t?"OK ( updated again!)":201===t?"OK ( created )":404===t?"Fail (Already submitted)":"Fail (code="+t.toString()+")"},toImageCoord:function(t,s){var e=this.viewRect,n=t/this.width*e.w+e.l,i=s/this.height*e.h+e.t;return new V(n,i)},fromImageCoord:function(t,s){var e=this.viewRect,n=(t-e.l)*this.width/e.w,i=(s-e.t)*this.height/e.h;return new V(n,i)},getClassChar:function(t){return this.clsChars.charAt(t)},onMouseLClick:function(t){console.log("onMouseLClick",t);var s=this.toImageCoord(t.offsetX,t.offsetY);if(!0!==this.isLabelCreating){if(!0!==t.altKey)if(this.labels.length>0){this.curLabel=null,this.curSide=null;for(var e=0;e<this.labels.length;e++)if(1==this.labels[e].include(s)){this.curLabel=this.labels[e];break}this.updateView()}else;else if(null!=this.curLabel){if(null!=this.curSide)0==this.curSide&&this.curLabel.tset(s.y),1==this.curSide&&this.curLabel.rset(s.x),2==this.curSide&&this.curLabel.bset(s.y),3==this.curSide&&this.curLabel.lset(s.x);else{var n=[Math.abs(s.y-this.curLabel.t()),Math.abs(s.x-this.curLabel.r()),Math.abs(s.y-this.curLabel.b()),Math.abs(s.x-this.curLabel.l())],i=n.indexOf(Math.min.apply(Math,n));0==i&&this.curLabel.tset(s.y,20),1==i&&this.curLabel.rset(s.x,20),2==i&&this.curLabel.bset(s.y,20),3==i&&this.curLabel.lset(s.x,20)}this.updateView()}}else{var l=this.creatingLabel;if(l.push(s),2==this.creatingLabel.length){var o=Math.min(l[0].x,l[1].x),a=Math.max(l[0].x,l[1].x),u=Math.min(l[0].y,l[1].y),c=Math.max(l[0].y,l[1].y),h=new j(0,(o+a)/2,(u+c)/2,a-o,c-u);this.labels.push(h),this.isLabelCreating=!1,this.creatingLabel=[],this.curLabel=h,this.curSide=0,this.sortLabels()}this.updateView()}},onKeyDown:function(t){if(!0!==this.isMouseProcessng)if(!0!==this.isLabelCreating){if("`"==t.key||"~"==t.key){var s=null,e=this.labels.indexOf(this.curLabel);if(e<0)s=0;else if(null==this.curSide)s=4*e;else{var n="`"==t.key?1:-1;s=4*e+this.curSide+n}return s=(s+4*this.labels.length)%(4*this.labels.length),this.curLabel=this.labels[Math.floor(s/4)],this.curSide=s%4,void this.updateView()}if("1"==t.key||"!"==t.key){var i=this.labels.indexOf(this.curLabel);if(i<0)i=0;else{var l="1"==t.key?1:-1;i+=l}return i=(i+this.labels.length)%this.labels.length,this.curLabel=this.labels[i],this.curSide=0,void this.updateView()}if("x"==t.key)return this.isShowCls=!this.isShowCls,void this.updateView();if("c"==t.key)return this.isLabelCreating=!0,this.creatingLabel=[],this.curLabel=null,void(this.curSide=null);if(null!=this.curLabel){if("Escape"==t.key&&(this.curLabel=null,this.curSide=null),"d"==t.key){var o=this.labels.indexOf(this.curLabel);o>=0&&(this.labels.splice(o,1),this.curSide=null)}return"ArrowLeft"==t.key&&(1==this.curSide&&this.curLabel.rset(this.curLabel.r()-1),3==this.curSide&&this.curLabel.lset(this.curLabel.l()-1)),"ArrowRight"==t.key&&(1==this.curSide&&this.curLabel.rset(this.curLabel.r()+1),3==this.curSide&&this.curLabel.lset(this.curLabel.l()+1)),"ArrowUp"==t.key&&(0==this.curSide&&this.curLabel.tset(this.curLabel.t()-1),2==this.curSide&&this.curLabel.bset(this.curLabel.b()-1)),"ArrowDown"==t.key&&(0==this.curSide&&this.curLabel.tset(this.curLabel.t()+1),2==this.curSide&&this.curLabel.bset(this.curLabel.b()+1)),this.sortLabels(),void this.updateView()}}else"Escape"==t.key&&(this.isLabelCreating=!1,this.creatingLabel=[],this.updateView());else"Escape"==t.key&&this.isMouseProcessing&&(this.isMouseProcessing=!1)},sortLabels:function(){this.labels.sort((function(t,s){return t.cx<s.cx?-1:t.cx>s.cx?1:0}))},drawLabel:function(t,s,e){var n=arguments.length>3&&void 0!==arguments[3]?arguments[3]:"red",i=arguments.length>4&&void 0!==arguments[4]?arguments[4]:"blue";4==t.length&&t.push(t[0].copy());var l=this;t=t.map((function(t){return l.fromImageCoord(t.x,t.y)}));var o=this.canvas;o.lineWidth="1";for(var a=0;a<t.length-1;a++)o.strokeStyle=e===a?"blue":n,o.beginPath(),o.moveTo(t[a].x,t[a].y),o.lineTo(t[a+1].x,t[a+1].y),o.stroke();o.fillStyle=i,o.fillRect(t[0].x-2,t[0].y-2,4,4),this.isShowCls&&(o.textAlign="left",o.textBaseline="top",o.font="28px malgun gothic",o.fillStyle="rgba(255, 0, 255, 180)",o.fillText(s,t[0].x+2,t[0].y+2))},drawCreatingLabel:function(t){var s=arguments.length>1&&void 0!==arguments[1]?arguments[1]:"blue",e=this.fromImageCoord(t[0].x,t[0].y),n=this.canvas;n.fillStyle=s,n.fillRect(e.x-2,e.y-2,4,4)},updateView:function(){var t=this;if(null!==this.canvas&&null!==this.image){var s=this.viewRect;this.width=s.w*this.scale*this.viewScale,this.height=s.h*this.scale*this.viewScale,this.$nextTick((function(){t.canvas.drawImage(t.image,s.l,s.t,s.w,s.h,0,0,t.width,t.height);var e,n=Object(v["a"])(t.labels);try{for(n.s();!(e=n.n()).done;){var i=e.value,l=i===t.curLabel?"yellow":"red",o=i===t.curLabel?t.curSide:null;t.drawLabel(i.pts(),t.getClassChar(i.cls),o,l)}}catch(a){n.e(a)}finally{n.f()}t.isLabelCreating&&t.drawCreatingLabel(t.creatingLabel)}))}},setViewRect:function(t,s,e,n){this.viewRect.l=t,this.viewRect.t=s,this.viewRect.w=e,this.viewRect.h=n},setCls:function(t){null!=this.curLabel&&(this.curLabel.cls=t,this.updateView()),this.setFocusToCanvas()},setFocusToCanvas:function(){document.getElementById("imgCanvas").focus()},onCls00:function(){this.setCls(0)},onCls01:function(){this.setCls(1)},onCls02:function(){this.setCls(2)},onCls03:function(){this.setCls(3)},onCls04:function(){this.setCls(4)},onCls05:function(){this.setCls(5)},onCls06:function(){this.setCls(6)},onCls07:function(){this.setCls(7)},onCls08:function(){this.setCls(8)},onCls09:function(){this.setCls(9)},onCls10:function(){this.setCls(10)},onCls11:function(){this.setCls(11)},onCls12:function(){this.setCls(12)},onCls13:function(){this.setCls(13)},onCls14:function(){this.setCls(14)},onCls15:function(){this.setCls(15)},onCls16:function(){this.setCls(16)},onCls17:function(){this.setCls(17)},onCls18:function(){this.setCls(18)},onCls19:function(){this.setCls(19)},onCls20:function(){this.setCls(20)},onCls21:function(){this.setCls(21)},onCls22:function(){this.setCls(22)},onCls23:function(){this.setCls(23)},onCls24:function(){this.setCls(24)},onCls25:function(){this.setCls(25)},onCls26:function(){this.setCls(26)},onCls27:function(){this.setCls(27)},onCls28:function(){this.setCls(28)},onCls29:function(){this.setCls(29)},onCls30:function(){this.setCls(30)},onCls31:function(){this.setCls(31)},onCls32:function(){this.setCls(32)},onCls33:function(){this.setCls(33)},onCls34:function(){this.setCls(34)},onCls35:function(){this.setCls(35)},onCls36:function(){this.setCls(36)},onCls37:function(){this.setCls(37)},onCls38:function(){this.setCls(38)},onCls39:function(){this.setCls(39)},onCls40:function(){this.setCls(40)},onCls41:function(){this.setCls(41)},onCls42:function(){this.setCls(42)},onCls43:function(){this.setCls(43)},onCls44:function(){this.setCls(44)},onCls45:function(){this.setCls(45)},onCls46:function(){this.setCls(46)},onCls47:function(){this.setCls(47)},onCls48:function(){this.setCls(48)},onCls49:function(){this.setCls(49)},onCls50:function(){this.setCls(50)},onCls51:function(){this.setCls(51)},onCls52:function(){this.setCls(52)},onCls53:function(){this.setCls(53)},onCls54:function(){this.setCls(54)},onCls55:function(){this.setCls(55)},onCls56:function(){this.setCls(56)},onCls57:function(){this.setCls(57)},onCls58:function(){this.setCls(58)},onCls59:function(){this.setCls(59)},onCls60:function(){this.setCls(60)},onCls61:function(){this.setCls(61)},onCls62:function(){this.setCls(62)},onCls63:function(){this.setCls(63)},onCls64:function(){this.setCls(64)},onCls65:function(){this.setCls(65)},onCls66:function(){this.setCls(66)}}},P=D,E=(e("1918"),Object(u["a"])(P,S,M,!1,null,"021d8441",null)),I=E.exports,J=function(){var t=this,s=t.$createElement;t._self._c;return t._m(0)},T=[function(){var t=this,s=t.$createElement,n=t._self._c||s;return n("div",{staticClass:"help"},[n("h4",[t._v("** Start 버튼 **")]),n("ul",[n("li",[t._v("새로 편집할 데이터를 불러옵니다.")])]),n("h4",[t._v("** Submit 버튼 **")]),n("ul",[n("li",[t._v("편집한 결과를 전송합니다. 한 번 submit 되면 다시 submit 불가")]),n("li",[t._v("혹시 잘못된 결과를 submit 했으면 저에게 file 명을 알려주세요")])]),n("h4",[t._v("** 편집 방법 **")]),n("ul",[n("li",[t._v("이미 적당한 label들이 존재하고, 요 점들을 적절히 옮겨주시면 됩니다.")]),n("li",[t._v("left-top이 첫번째 점(파란 점)이 되어야 하고 시계 방향 순서입니다.")]),n("li",[t._v("번호판의 전체 (글자 + 외곽)이 적당히 포함되도록 해주세요")]),n("li",[n("b",[t._v("하다가 버그 등으로 인해 꼬이면 그냥 새로고침 하시면 됩니다.")])])]),n("img",{attrs:{src:e("233b")}}),n("h4",[t._v("** 조작 방법 **")]),n("ul",[n("li",[n("b",[t._v("화면 확대")]),t._v(" : 마우스로 적당히 드래그앤드랍")]),n("li",[n("b",[t._v("화면 확대 취소")]),t._v(" : '`' 키. (1 왼쪽의 backquote 키)")]),n("li",[n("b",[t._v("label 선택")]),t._v(" : label 안쪽 영역을 더블클릭. 선택되면 색이 노란색으로 바뀝니다.")]),n("li",[n("b",[t._v("label 선택취소")]),t._v(" : Esc 키 혹은 다시 더블클릭")]),n("li",[n("b",[t._v("점 옮기기")]),t._v(" : 선택된 상태에서 클릭하면 현재 가장 가까운 점이 클릭한 위치로 올겨집니다.")]),n("li",[n("b",[t._v("label 추가")]),t._v(" : 미선택 상태에서 c 키를 누르고 네 점을 클릭")]),n("li",[n("b",[t._v("label 삭제")]),t._v(" : 선택 상태에서 d 키")])])])}],B={name:"Help",props:{msg:String}},$=B,q=(e("79c8"),Object(u["a"])($,J,T,!1,null,"6d36dbd1",null)),A=q.exports;n["a"].use(r["a"]);var K=new r["a"]({mode:"history",routes:[{path:"/",name:"wpodanno",component:x},{path:"/ocranno",name:"ocranno",component:I},{path:"/help",name:"help",component:A}]}),N=e("bc3a"),X=e.n(N);n["a"].config.productionTip=!1,n["a"].prototype.$http=X.a,new n["a"]({render:function(t){return t(h)},router:K}).$mount("#app")},"59a5":function(t,s,e){"use strict";var n=e("7b73"),i=e.n(n);i.a},"79c8":function(t,s,e){"use strict";var n=e("43d0"),i=e.n(n);i.a},"7b73":function(t,s,e){},"85ec":function(t,s,e){},cc9a:function(t,s,e){}});
//# sourceMappingURL=app.f9af2806.js.map