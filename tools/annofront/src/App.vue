<template>
  <v-app>
    <v-navigation-drawer
      v-model="drawer"
      app
    >
      <v-list dense>

        <v-list-item link to="/">
          <v-list-item-action>
            <v-icon>mdi-home</v-icon>
          </v-list-item-action>
          <v-list-item-content>
            <v-list-item-title>home</v-list-item-title>
          </v-list-item-content>
        </v-list-item>

        <v-list-item link to="/wpodanno">
          <v-list-item-action>
            <v-icon>crop_7_5</v-icon>
          </v-list-item-action>
          <v-list-item-content>
            <v-list-item-title>Wpod Annotation</v-list-item-title>
          </v-list-item-content>
        </v-list-item>

        <v-list-item link to="/ocranno">
          <v-list-item-action>
            <v-icon>text_format</v-icon>
          </v-list-item-action>
          <v-list-item-content>
            <v-list-item-title>Ocr Annotation</v-list-item-title>
          </v-list-item-content>
        </v-list-item>

      </v-list>
    </v-navigation-drawer>

    <v-app-bar
      app
      color="primary"
      dark
    >
      <v-app-bar-nav-icon @click.stop="drawer = !drawer"></v-app-bar-nav-icon>
      <v-toolbar-title>VDMS annotation tool</v-toolbar-title>

      <v-spacer></v-spacer>

      <v-btn
        href="http://victech.io/"
        target="_blank"
        text
      >
        <span class="mr-2">victech</span>
        <v-icon>mdi-open-in-new</v-icon>
      </v-btn>
    </v-app-bar>

    <!-- <v-main>
      <HelloWorld/>
    </v-main> -->

    <v-main>
        <router-view></router-view>
    </v-main>
  </v-app>
</template>

<script>

export default {
  name: 'App',

  components: {
  },

  data: () => ({
    drawer: null,
  }),
};
</script>
