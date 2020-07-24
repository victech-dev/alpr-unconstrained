<template>
  <div class="help">
    <h4>** Start 버튼 **</h4>
    <ul>
      <li>새로 편집할 데이터를 불러옵니다.</li>
    </ul>
    <h4>** Submit 버튼 **</h4>
    <ul>
      <li>편집한 결과를 전송합니다. 한 번 submit 되면 다시 submit 불가</li>
      <li>혹시 잘못된 결과를 submit 했으면 저에게 file 명을 알려주세요</li>
    </ul>
    <h4>** 편집 방법 **</h4>
    <ul>
      <li>이미 적당한 label들이 존재하고, 요 점들을 적절히 옮겨주시면 됩니다.</li>
      <li>left-top이 첫번째 점(파란 점)이 되어야 하고 시계 방향 순서입니다.</li>
      <li>번호판의 전체 (글자 + 외곽)이 적당히 포함되도록 해주세요</li>
    </ul>
    <img src="../assets/sample.png" />
    <h4>** 조작 방법 **</h4>
    <ul>
      <li><b>화면 확대</b> : 마우스로 적당히 드래그앤드랍</li>
      <li><b>화면 확대 취소</b> : '`' 키. (1 왼쪽의 backquote 키)</li>
      <li><b>label 선택</b> : label 안쪽 영역을 더블클릭. 선택되면 색이 노란색으로 바뀝니다.</li>
      <li><b>label 선택취소</b>Esc 키 혹은 다시 더블클릭</li>
      <li><b>점 옮기기</b> : 선택된 상태에서 클릭하면 현재 가장 가까운 점이 클릭한 위치로 올겨집니다.</li>
      <li><b>label 추가</b> : 미선택 상태에서 c 키를 누르고 네 점을 클릭</li>
      <li><b>label 삭제</b> : 선택 상태에서 d 키</li>
    </ul>
  </div>
</template>

<script>
export default {
  name: 'Help',
  props: {
    msg: String
  }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
h4 {
  margin: 40px 0 0;
}
ul {
  list-style-type: none;
  padding: 0;
}
/* li {
  display: inline-block;
  margin: 0 10px;
} */
a {
  color: #42b983;
}
</style>
