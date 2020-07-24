<template>
  <div id="app">
    <p>
      <!-- 네비게이션을 위해 router-link 컴포넌트를 사용합니다. -->
      <!-- 구체적인 속성은 `to` prop을 이용합니다. -->
      <!-- 기본적으로 `<router-link>`는 `<a>` 태그로 렌더링됩니다.-->
      <router-link to="/" tag="button">Editor</router-link>--
      <router-link to="/help" tag="button">Help</router-link>
    </p>
    <!-- <img alt="Vue logo" src="./assets/logo.png"> -->
    <!-- <HelloWorld msg="Welcome to Your Vue.js App"/> -->
    <router-view></router-view>
    
  </div>
</template>

<script>

export default {
  name: 'App',
  components: {
  }
}
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}
</style>
