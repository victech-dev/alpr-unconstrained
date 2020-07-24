module.exports = { 
    devServer: {
        proxy: { 
            '/annos': { 
                target: 'http://localhost:3000/annos',
                changeOrigin: true, 
                pathRewrite: { 
                    '^/annos': ''
                } 
            } 
        } 
    },
    outputDir: '../annoback/public',
}