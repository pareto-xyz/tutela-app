import React from 'react';
import { Table } from 'react-bootstrap';
import MyTooltip from './MyTooltip';

/**
 * 
 * @param {*} keyValues is an object of data to be displayed. it may contain a "hovers" object that will add hover info. 
 * @returns 
 */
export default function AgnosticTable({ toIgnore = new Set(), keyValues, aliases = {} }) {
    let hovers = null;
    if (typeof (keyValues.hovers) === 'object') {
        hovers = keyValues.hovers;
    }

    return (
        <Table hover size='sm' className="detail-table">
            <tbody className="body-table">
                {Object.entries(keyValues).map((entry, idx) => {
                    let [k, value] = entry;
                    const ogK = k; //original key
                    if (k === 'hovers' || toIgnore.has(k) || typeof (k) === 'object' || value === null) return null;
                    if (aliases[k]) {
                        k = aliases[k];
                    }
                    if (aliases[value]) {
                        value = aliases[value];
                    }
                    return (
                        <tr className="detail-row" key={idx}>
                            <td>{k}
                                {hovers && hovers[ogK] && <MyTooltip tooltipText={hovers[ogK]} />}:
                            </td>
                            <td>{typeof (value) === 'object'
                                ? <AgnosticTable keyValues={value} aliases={aliases} />
                                : value}</td>
                        </tr>)
                }
                )}
            </tbody>
        </Table>
    )
}
