import React from 'react';
import { Form } from 'react-bootstrap';

export default function Pagination({ paginationData, getNewResults }) {
    const { cluster_size, limit, num_pages, page } = paginationData;
    const firstInRange = page * limit + 1;
    const lastInRange = Math.min(cluster_size, (page + 1) * limit);

    const changePage = desiredPage => { //zero-indexed
        getNewResults({page: desiredPage});
    }

    return (
        <div className='all-pagination'>
            <div>
                results {firstInRange}-{lastInRange} out of {cluster_size}
            </div>
            <div className="pagination">
                <button onClick={e => {
                    if (page === 0) return;
                    changePage(page-1);
                }} 
                className="page-button" id="prev-page" aria-label="Previous">
                    <span aria-hidden="true">&laquo;</span>
                </button>
                page
                <Form.Select id="page-number" className="select-page" onChange={e => changePage(e.target.value)}>
                    {Array(num_pages).fill(0).map((_, i) => <option value={i+1}>{i+1}</option>)}
                </Form.Select>
                out of {num_pages}
                <button onClick={e => {
                    if (page >= num_pages - 1) return;
                    changePage(page + 1);
                 }} 
                 className="page-button" id="next-page" aria-label="Next">
                    <span aria-hidden="true">&raquo;</span>
                </button>
            </div>
        </div>
    );
}