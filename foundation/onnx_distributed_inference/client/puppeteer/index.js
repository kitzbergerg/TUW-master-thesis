import puppeteer from 'puppeteer';

async function runWebApp() {
    const browser = await puppeteer.launch({
        headless: 'new',
        args: [
            '--no-sandbox',
            '--headless=new',
            '--use-angle=vulkan',
            '--enable-features=Vulkan',
            '--disable-vulkan-surface',
            '--enable-unsafe-webgpu',
        ]
    });

    const page = await browser.newPage();

    page.on('console', (msg) => {
        const type = msg.type();
        console.log(`[Browser Console ${type}] ${msg.text()}`);
    });
    page.on('pageerror', (error) => {
        console.error(`[Browser JS Error] ${error.message}`);
    });
    page.on('requestfailed', (request) => {
        console.error(`[Browser Network Error] ${request.url()} failed: ${request.failure().errorText}`);
    });

    await page.goto('http://localhost:8080/');

    console.log('Browser connected and running. Press Ctrl+C to exit.');


    process.on('SIGINT', async () => {
        console.log('Received SIGINT (Ctrl+C). Closing browser and exiting...');
        await browser.close();
        process.exit(0);
    });
    process.on('SIGTERM', async () => {
        console.log('Received SIGTERM. Closing browser and exiting...');
        await browser.close();
        process.exit(0);
    });

    process.on('uncaughtException', async (err) => {
        console.error('Uncaught exception:', err);
        await browser.close();
        process.exit(1);
    });

}

runWebApp().catch(async (error) => {
    console.error('Error in main process:', error);
    process.exit(1);
});