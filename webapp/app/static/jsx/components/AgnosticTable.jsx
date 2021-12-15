import React from 'react';
import { Table } from 'react-bootstrap';

export default function AgnosticTable({ toIgnore = new Set(), keyValues, aliases = {} }) {
    return (
        <Table hover size='sm' className="detail-table">
            <tbody>
                {keyValues.map((entry, idx) => {
                    let [k, value] = entry;
                    if (toIgnore.has(k) || typeof(k) === 'object' || value === null) return < ></>;
                    if (aliases[k]) {
                        k = aliases[k];
                    }
                    if (aliases[value]) {
                        value = aliases[value];
                    }
                    return (
                        <tr className="detail-row" key={idx}>
                            <td>{k}</td>
                            <td>{typeof(value) === 'object' ? <AgnosticTable keyValues={Object.entries(value)} aliases={aliases} /> : value}</td>
                        </tr>)
                }
                )}
            </tbody>
        </Table>
    )
}
