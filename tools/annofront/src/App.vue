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
      <!-- <div class="d-flex align-center">
        <v-img
          alt="Vuetify Name"
          class="shrink mt-1 hidden-sm-and-down"
          contain
          min-width="100"
          src="http://victech.io/images/VIC_Logo.png"
          width="120"
        />
      </div> -->

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

    <v-content>
        <router-view></router-view>
    </v-content>
  </v-app>
</template>

<script>
//import HelloWorld from './components/HelloWorld';

export default {
  name: 'App',

  components: {
    //HelloWorld,
  },

  data: () => ({
    drawer: null,
  }),
};
</script>
