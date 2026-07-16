package pgqueue

import (
	"context"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/skerkour/stdx-go/uuid"
)

func (queue *PostgreSQLQueue) Push(ctx context.Context, querier Querier, newJob NewJobInput) (err error) {
	now := time.Now().UTC()

	job, err := validateJob(now, newJob)
	if err != nil {
		return err
	}

	_, err = querier.Exec(ctx, `INSERT INTO queue
		(id, created_at, updated_at, scheduled_for, failed_attempts, status, type, data, retry_max, retry_delay, retry_strategy, timeout)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)`,
		job.ID, job.CreatedAt, job.UpdatedAt, job.ScheduledFor, job.FailedAttempts,
		job.Status, job.Type, job.DataJson, job.RetryMax, job.RetryDelay,
		job.RetryStrategy, job.Timeout)
	if err != nil {
		return fmt.Errorf("pgqueue: inserting job: %w", err)
	}

	return nil
}

func (queue *PostgreSQLQueue) PushMany(ctx context.Context, querier Querier, newJobs []NewJobInput) error {
	if len(newJobs) == 0 {
		return nil
	}

	now := time.Now().UTC()

	rows := make([][]any, len(newJobs))
	for i, newJob := range newJobs {
		job, err := validateJob(now, newJob)
		if err != nil {
			return fmt.Errorf("pgqueue: validating job %d: %w", i, err)
		}
		rows[i] = []any{
			job.ID, job.CreatedAt, job.UpdatedAt, job.ScheduledFor, job.FailedAttempts, job.Status,
			job.Type, job.DataJson, job.RetryMax, job.RetryDelay, job.RetryStrategy, job.Timeout,
		}
	}

	_, err := querier.CopyFrom(ctx,
		pgx.Identifier{"queue"},
		[]string{
			"id", "created_at", "updated_at", "scheduled_for", "failed_attempts", "status",
			"type", "data", "retry_max", "retry_delay", "retry_strategy", "timeout",
		},
		pgx.CopyFromRows(rows),
	)
	if err != nil {
		return fmt.Errorf("pgqueue: CopyFrom inserting jobs: %w", err)
	}

	return nil
}

func (q *PostgreSQLQueue) Pull(ctx context.Context, numberOfJobs uint64) ([]Job, error) {
	now := time.Now().UTC()
	query := `UPDATE queue
	SET status = $1, updated_at = $2
	WHERE id IN (
		SELECT id
		FROM queue
		WHERE status = $3 AND scheduled_for <= $4 AND failed_attempts <= queue.retry_max
		ORDER BY scheduled_for
		FOR UPDATE SKIP LOCKED
		LIMIT $5
	)
	RETURNING *`

	// If we don't find jobs, we continue to make requests, either until at least one job is found
	// or until approximately 1 second has elapsed.
	// This technique is known as long polling.
	pollingTimer := time.NewTimer(0) // fires immediately for first iteration
	for i := 0; i < 10; i++ {
		select {
		case <-ctx.Done():
			return nil, nil
		case <-pollingTimer.C:
		}

		rows, err := q.pool.Query(ctx, query,
			JobStatusRunning, now, JobStatusQueued, now, int64(numberOfJobs))
		if err != nil {
			return nil, fmt.Errorf("pgqueue: pulling jobs: %w", err)
		}

		jobs, err := pgx.CollectRows(rows, pgx.RowToStructByName[Job])
		if err != nil {
			return nil, fmt.Errorf("pgqueue: scanning pulled jobs: %w", err)
		}

		if len(jobs) > 0 {
			return jobs, nil
		}
		pollingTimer.Reset(100 * time.Millisecond)
	}

	return nil, nil
}

func (queue *PostgreSQLQueue) DeleteJob(ctx context.Context, jobID uuid.UUID) error {
	_, err := queue.pool.Exec(ctx, "DELETE FROM queue WHERE id = $1", jobID)
	if err != nil {
		return fmt.Errorf("pgqueue: deleting job: %w", err)
	}
	return nil
}

func (q *PostgreSQLQueue) FailJob(ctx context.Context, jobID uuid.UUID) error {
	now := time.Now().UTC()

	_, err := q.pool.Exec(ctx, `UPDATE queue
	SET
		status = CASE
			WHEN failed_attempts + 1 >= retry_max THEN $1
			ELSE $2
		END,
		updated_at = $3,
		failed_attempts = failed_attempts + 1,
		scheduled_for = CASE
			WHEN failed_attempts + 1 >= retry_max THEN scheduled_for
			ELSE $3 + (retry_delay * CASE WHEN retry_strategy = $4 THEN failed_attempts + 1 ELSE 1 END) * INTERVAL '1 second'
		END
	WHERE id = $5`,
		JobStatusFailed, JobStatusQueued, now, RetryStrategyExponential, jobID)
	if err != nil {
		return fmt.Errorf("pgqueue: failing job: %w", err)
	}
	return nil
}

func (queue *PostgreSQLQueue) Clear(ctx context.Context) error {
	_, err := queue.pool.Exec(ctx, "DELETE FROM queue")
	if err != nil {
		return fmt.Errorf("pgqueue: clearing queue: %w", err)
	}
	return nil
}

func (q *PostgreSQLQueue) GetJob(ctx context.Context, jobID uuid.UUID) (Job, error) {
	rows, err := q.pool.Query(ctx, "SELECT * FROM queue WHERE id = $1", jobID)
	if err != nil {
		return Job{}, fmt.Errorf("pgqueue: getting job: %w", err)
	}

	return pgx.CollectExactlyOneRow(rows, pgx.RowToStructByName[Job])
}

func (q *PostgreSQLQueue) GetFailedJobs(ctx context.Context, limit int64) ([]Job, error) {
	rows, err := q.pool.Query(ctx, `SELECT * FROM queue WHERE status = $1 ORDER BY created_at DESC LIMIT $2`,
		JobStatusFailed, limit)
	if err != nil {
		return nil, fmt.Errorf("pgqueue: querying failed jobs: %w", err)
	}

	return pgx.CollectRows(rows, pgx.RowToStructByName[Job])
}
