/**
     * checks if the addr is a valid ethereum address (hex and 42 char long including the 0x) 
     * @param {string} addr 
     */
function isValid(addr) {
    const re = /[0-9A-Fa-f]{40}/g;
    addr = addr.trim();
    const ans = ((addr.substr(0, 2) === '0x' && re.test(addr.substr(2))) //with the 0x
        || (re.test(addr))); //without the 0x
    return ans;
}

/**
 * query is a map. 
 * returns query string including the ? at the beginning 
 * @param {string} query 
 */
function buildQueryString(query) {
    let queryString = '?';
    for (const [key, val] of Object.entries(query)) {
        queryString += key + '=' + val + '&';
    }
    if (queryString[queryString.length - 1] === '&') {
        queryString = queryString.substr(0, queryString.length - 1);
    }
    return queryString;
}

export { isValid, buildQueryString }