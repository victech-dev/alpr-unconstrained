module.exports = { 
    devServer: {
        proxy: { 
            '/wpodannos': { 
                target: 'http://localhost:3000/wpodannos',
                changeOrigin: true, 
                pathRewrite: { 
                    '^/wpodannos': ''
                } 
            } 
        } 
    },
    outputDir: '../annoback/public',
}