#include "Bot.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "NeuralBot.h"
#include "Arena.h"
#include <math.h>

#define RELOAD_TIME

__device__ void BotUpdate(SNeuralBot b, SNeuralNetwork nnwrk, SArena a) {
	b.reload--;
	BotAction ba = GetBotAction(b, nnwrk);

	if (ba.acceleration > 0) {
		b.vel += ACCELERATION;
	}
	else if (ba.acceleration < 0) {
		b.vel -= ACCELERATION;
		if (b.vel <= 0) b.vel = 0;
	}
	if (ba.rotation > 0) {
		b.rot = fmodf(b.rot + ROTATION_SPEED, M_2PI);
	}
	else if (ba.rotation < 0) {
		b.rot = fmodf(b.rot - ROTATION_SPEED + M_2PI, M_2PI);
	}

	if (ba.shoot &&  b.reload <= 0 && b.ammo > 0) {
		b.ammo--;
		b.reload = RELOAD;
		AShootFrom(a, b.posx, b.posy);
	}

	b.posx += GetVelX(b);
	b.posy += GetVelY(b);
}

__device__ float GetVelX(SNeuralBot b) {
	return b.vel * cos(b.rot);
}

__device__ float GetVelY(SNeuralBot b) {
	return b.vel * sin(b.rot);
}

__device__ int BotCollides(SNeuralBot bo, SBullet bu) {
	float dist = pointDistance(bo.posx + BOT_RADIUS, bo.posy + BOT_RADIUS, bu.posx, bu.posy, bu.prevPosx, bu.prevPosy);
	return dist <= (BOT_RADIUS + BULLET_RADIUS) / 2;
}
__device__ int BotCollides(SNeuralBot bo, SAmmo a) {
	float dist = pointDistance(bo.posx + BOT_RADIUS, bo.posy + BOT_RADIUS, a.posx, a.posy, bo.posx + GetVelX(bo), bo.posy + GetVelY(bo));
	return dist <= (BOT_RADIUS + AMMO_RADIUS) / 2;
}

__device__ float pointDistance(float x, float y, float x1, float y1, float x2, float y2) {

	float A = x - x1;
	float B = y - y1;
	float C = x2 - x1;
	float D = y2 - y1;

	float dot = A * C + B * D;
	float len_sq = C * C + D * D;
	float param = -1;
	if (len_sq != 0) //in case of 0 length line
		param = dot / len_sq;

	float xx, yy;

	if (param < 0) {
		xx = x1;
		yy = y1;
	}
	else if (param > 1) {
		xx = x2;
		yy = y2;
	}
	else {
		xx = x1 + param * C;
		yy = y1 + param * D;
	}

	float dx = x - xx;
	float dy = y - yy;
	return sqrt(dx * dx + dy * dy);
}