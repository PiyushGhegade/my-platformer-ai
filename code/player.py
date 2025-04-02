import pygame 
from support import import_folder
from math import sin
from ai import AI
import sys
from settings import control_ai

class Player(pygame.sprite.Sprite):
	def __init__(self,pos,surface,create_jump_particles,change_health):
		super().__init__()
		self.import_character_assets()
		self.frame_index = 0
		self.animation_speed = 0.15
		self.image = self.animations['idle'][self.frame_index]
		self.rect = self.image.get_rect(topleft = pos)
		
		# Add these velocity tracking attributes
		self.previous_pos = pos
		self.velocity_y = 0
		self.velocity_x = 0
		self.previous_y = pos[1]  # Track previous y position for velocity calculation
		self.max_fall_velocity = 20  # Optional: cap maximum falling speed

		# dust particles 
		self.import_dust_run_particles()
		self.dust_frame_index = 0
		self.dust_animation_speed = 0.15
		self.display_surface = surface
		self.create_jump_particles = create_jump_particles

		# player movement
		self.direction = pygame.math.Vector2(0,0)
		self.speed = 8
		self.gravity = 0.8
		self.jump_speed = -16
		self.collision_rect = pygame.Rect(self.rect.topleft,(50,self.rect.height))

		# player status
		self.status = 'idle'
		self.facing_right = True
		self.on_ground = False
		self.on_ceiling = False
		self.on_left = False
		self.on_right = False

		# health management
		self.change_health = change_health
		self.invincible = False
		self.invincibility_duration = 500
		self.hurt_time = 0

		# audio 
		self.jump_sound = pygame.mixer.Sound('../audio/effects/jump.wav')
		self.jump_sound.set_volume(0.5)
		self.hit_sound = pygame.mixer.Sound('../audio/effects/hit.wav')

	def import_character_assets(self):
		character_path = '../graphics/character/'
		self.animations = {'idle':[],'run':[],'jump':[],'fall':[]}

		for animation in self.animations.keys():
			full_path = character_path + animation
			self.animations[animation] = import_folder(full_path)

	def import_dust_run_particles(self):
		self.dust_run_particles = import_folder('../graphics/character/dust_particles/run')

	def animate(self):
		animation = self.animations[self.status]

		# loop over frame index 
		self.frame_index += self.animation_speed
		if self.frame_index >= len(animation):
			self.frame_index = 0

		image = animation[int(self.frame_index)]
		if self.facing_right:
			self.image = image
			self.rect.bottomleft = self.collision_rect.bottomleft
		else:
			flipped_image = pygame.transform.flip(image,True,False)
			self.image = flipped_image
			self.rect.bottomright = self.collision_rect.bottomright

		if self.invincible:
			alpha = self.wave_value()
			self.image.set_alpha(alpha)
		else:
			self.image.set_alpha(255)

		self.rect = self.image.get_rect(midbottom = self.rect.midbottom)		

	def run_dust_animation(self):
		if self.status == 'run' and self.on_ground:
			self.dust_frame_index += self.dust_animation_speed
			if self.dust_frame_index >= len(self.dust_run_particles):
				self.dust_frame_index = 0

			dust_particle = self.dust_run_particles[int(self.dust_frame_index)]

			if self.facing_right:
				pos = self.rect.bottomleft - pygame.math.Vector2(6,10)
				self.display_surface.blit(dust_particle,pos)
			else:
				pos = self.rect.bottomright - pygame.math.Vector2(6,10)
				flipped_dust_particle = pygame.transform.flip(dust_particle,True,False)
				self.display_surface.blit(flipped_dust_particle,pos)

	def get_input(self, ai_action=control_ai):
		"""Handles player input: AI-controlled or keyboard-based"""
		
		if ai_action is not None:
			# AI-based control
			if ai_action == 0:  # Move Left
				self.direction.x = -1
				self.facing_right = False
			elif ai_action == 1:  # Move Right
				self.direction.x = 1
				self.facing_right = True
			elif ai_action == 2 and self.on_ground:  # Jump
				self.jump()
				self.create_jump_particles(self.rect.midbottom)
			else:
				self.direction.x = 0  # Stop moving
		else:
			# Default to keyboard control
			keys = pygame.key.get_pressed()
			if keys[pygame.K_RIGHT]:
				self.direction.x = 1
				self.facing_right = True
			elif keys[pygame.K_LEFT]:
				self.direction.x = -1
				self.facing_right = False
			else:
				self.direction.x = 0

			if keys[pygame.K_SPACE] and self.on_ground:
				self.jump()
				self.create_jump_particles(self.rect.midbottom)

		# keys = pygame.key.get_pressed()
		# ai_instance = AI()
		# random_number = ai_instance.get_random_number()

		# if True:
		# 	self.direction.x = 1
		# 	self.facing_right = True
		# elif False:
		# 	self.direction.x = -1
		# 	self.facing_right = False
		# else:
		# 	self.direction.x = 0

		# if random_number == 1 and self.on_ground:
		# 	self.jump()
		# 	self.create_jump_particles(self.rect.midbottom)
		"""Move the player based on AI action"""

		# if action == 1:  # Move Right
		# 	self.direction.x = 1
		# 	self.facing_right = True
		# elif action == 0:  # Move Left
		# 	self.direction.x = -1
		# 	self.facing_right = False
		# else:  # No movement
		# 	self.direction.x = 0

		# if action == 2 and self.on_ground:  # Jump
		# 	self.jump()
		# 	self.create_jump_particles(self.rect.midbottom)

	def get_status(self):
		if self.direction.y < 0:
			self.status = 'jump'
		elif self.direction.y > 1:
			self.status = 'fall'
		else:
			if self.direction.x != 0:
				self.status = 'run'
			else:
				self.status = 'idle'

	def apply_gravity(self):
		self.previous_y = self.collision_rect.y  # Store position before applying gravity
		self.direction.y += self.gravity
		self.collision_rect.y += self.direction.y
    
    # Update velocity_y (pixels per frame)
		self.velocity_y = self.collision_rect.y - self.previous_y
    
    # Optional: Cap maximum falling velocity
		if self.velocity_y > self.max_fall_velocity:
			self.velocity_y = self.max_fall_velocity
			self.direction.y = self.max_fall_velocity

	def jump(self):
		self.direction.y = self.jump_speed
		self.velocity_y = self.jump_speed  # Set initial jump velocity
		self.jump_sound.play()

	def get_damage(self):
		if not self.invincible:
			self.hit_sound.play()
			self.change_health(-10)
			self.invincible = True
			self.hurt_time = pygame.time.get_ticks()

	def _update_velocities(self):
		"""Calculate pixel-per-frame velocities"""
		current_x, current_y = self.rect.x, self.rect.y
		self.velocity_x = current_x - self.previous_pos[0]
		self.velocity_y = current_y - self.previous_pos[1]
		self.previous_pos = (current_x, current_y)

	def invincibility_timer(self):
		if self.invincible:
			current_time = pygame.time.get_ticks()
			if current_time - self.hurt_time >= self.invincibility_duration:
				self.invincible = False

	def wave_value(self):
		value = sin(pygame.time.get_ticks())
		if value >= 0: return 255
		else: return 0

	def get_velocity(self):
		"""Returns the player's current y-velocity"""
		return self.velocity_y,self.velocity_x

	def update(self):
		self.get_input()
		self.get_status()
		self.animate()
		self.run_dust_animation()
		self.invincibility_timer()
		self.wave_value()
		self._update_velocities()
	
	

	


		