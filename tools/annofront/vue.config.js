module.exports = { 
    devServer: {
        proxy: { 
            '/wpodannos': { 
                target: 'http://localhost:3000/wpodannos',
                changeOrigin: true, 
                pathRewrite: { 
                    '^/wpodannos': ''
                } 
            },
            '/ocrannos': { 
                target: 'http://localhost:3000/ocrannos',
                changeOrigin: true, 
                pathRewrite: { 
                    '^/ocrannos': ''
                } 
            } 
        } 
    },
    outputDir: '../annoback/public',
}