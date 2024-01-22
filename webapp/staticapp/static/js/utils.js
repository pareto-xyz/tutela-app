/**
 * checks if the addr is a valid ethereum address (hex and 42 char long including the 0x) 
 * @param {string} addr 
 */
function isValid(addr) {
    const re = /[0-9A-Fa-f]{40}/g;
    addr = addr.trim();
    const ans = ((addr.substr(0, 2) === '0x' && re.test(addr.substr(2))) //with the 0x
        || (re.test(addr))
        || addr.endsWith('.eth')); //without the 0x
    return ans;
}

export { isValid }